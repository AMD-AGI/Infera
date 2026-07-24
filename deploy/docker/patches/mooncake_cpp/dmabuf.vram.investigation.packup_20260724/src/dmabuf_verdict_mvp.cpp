// Reordered probe: occupy 150G BEFORE registering the 100G pool, to test whether
// dmabuf registration needs EXTRA physical memory. If reg needs extra physical
// VRAM, then with only ~38G free (100G pool + 150G occupier out of 288), the
// 100G registration must FAIL or be limited. If reg only pins already-present
// memory (no extra physical), it should SUCCEED even with ~38G free.
//
// Flow:
//   P1: malloc 100G pool
//   P2: malloc 150G occupier (kept alive)   -> physical ~250G used, ~38G free
//   P3: register the 100G pool (dmabuf|bare) -> OBSERVE success/fail + gauges
//   P4: free occupier, P5: dereg, P6: free pool
// Handshake-synced TTM sampling (sig dir); ground truth = reg success/fail.
// Usage: mvp <mode>   (dmabuf|bare)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>

static double G(long b){ return b/1073741824.0; }
static size_t hipFreeB(){ size_t f=0,t=0; (void)hipMemGetInfo(&f,&t); return f; }
static long rd(const char*f){ FILE*p=fopen(f,"r"); long v=-1; if(p){ if(fscanf(p,"%ld",&v)!=1)v=-1; fclose(p);} return v; }

#define SIG "/mnt/vast/c_huggingface/sig"
static void phase(const char*tag){
  double hf=G((long)hipFreeB());
  double vr=G(rd("/sys/class/drm/card1/device/mem_info_vram_used"));
  double gt=G(rd("/sys/class/drm/card1/device/mem_info_gtt_used"));
  printf(">>> %s  hip_free=%.2f sysfs_vram=%.2f gtt=%.2f  (waiting host TTM sample)\n",tag,hf,vr,gt);
  fflush(stdout);
  char pf[256]; snprintf(pf,sizeof pf,"%s/PHASE",SIG);
  FILE*f=fopen(pf,"w"); if(f){ fprintf(f,"%s hip_free=%.2f sysfs_vram=%.2f gtt=%.2f\n",tag,hf,vr,gt); fclose(f); }
  char gf[256]; snprintf(gf,sizeof gf,"%s/GO",SIG);
  for(int i=0;i<200;i++){ if(access(gf,F_OK)==0){ unlink(gf); unlink(pf); return; } usleep(100000); }
  fprintf(stderr,"  (timeout GO %s)\n",tag); unlink(pf);
}

static ibv_pd* open_ionic(){ int n=0; ibv_device**l=ibv_get_device_list(&n); ibv_device*d=nullptr;
  for(int i=0;i<n;i++){const char*nm=ibv_get_device_name(l[i]); if(nm&&strstr(nm,"ionic")){d=l[i];break;}}
  if(!d&&n)d=l[0]; if(!d)return nullptr; ibv_context*c=ibv_open_device(d); return c?ibv_alloc_pd(c):nullptr; }
static int g_access=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
static ibv_mr* reg_one(ibv_pd*pd,const std::string&mode,float*p,size_t sz,int*out_fd){
  *out_fd=-1;
  if(mode=="dmabuf"){ hipDeviceptr_t base=nullptr; size_t asz=0; (void)hipMemGetAddressRange(&base,&asz,(hipDeviceptr_t)p);
    int fd=-1; uint64_t off=0; if(hsa_amd_portable_export_dmabuf((void*)base,asz,&fd,&off)!=HSA_STATUS_SUCCESS){ printf("    export FAIL\n"); return nullptr; }
    uint64_t roff=(uintptr_t)p-(uintptr_t)base+off; *out_fd=fd; return ibv_reg_dmabuf_mr(pd,roff,sz,(uintptr_t)p,fd,g_access); }
  return ibv_reg_mr(pd,p,sz,g_access);
}

int main(int argc,char**argv){
  std::string mode=argc>1?argv[1]:"dmabuf";
  hipSetDevice(0); hsa_init();
  ibv_pd*pd=open_ionic(); if(!pd){ printf("no pd\n"); return 2; }
  printf("### mode=%s REORDERED: occupy 150G BEFORE registering 100G ###\n",mode.c_str());
  phase("P0_empty");

  // P1: 100G pool as 100x1G
  const int NB=100; size_t bsz=(size_t)1*1073741824;
  std::vector<float*> pool;
  for(int i=0;i<NB;i++){ float*p=nullptr; if(hipMalloc(&p,bsz)!=hipSuccess){printf("pool malloc stop @%d\n",i);break;} (void)hipMemsetD8((hipDeviceptr_t)p,0,bsz); pool.push_back(p);} (void)hipDeviceSynchronize();
  printf("[pool: %zu x 1G]\n",pool.size());
  phase("P1_pool_100G");

  // P2: occupy 150G FIRST (kept alive) -> physical ~250G used
  size_t occSz=(size_t)150*1073741824; float* occ=nullptr;
  hipError_t oe=hipMalloc(&occ,occSz);
  if(oe==hipSuccess){ (void)hipMemsetD8((hipDeviceptr_t)occ,5,occSz); (void)hipDeviceSynchronize(); printf("[occupier 150G malloc: SUCCESS]\n"); }
  else { printf("[occupier 150G malloc: FAILED %s]\n",hipGetErrorString(oe)); }
  phase("P2_occupier_150G");

  // P3: NOW register the 100G pool with only ~38G physical free. Does reg succeed?
  int okreg=0; std::vector<ibv_mr*> mrs; std::vector<int> fds; int firstErr=0;
  for(size_t i=0;i<pool.size();i++){ int fd=-1; ibv_mr*mr=reg_one(pd,mode,pool[i],bsz,&fd);
    if(mr){ mrs.push_back(mr); fds.push_back(fd); okreg++; }
    else { if(!firstErr){firstErr=errno; printf("    reg FAILED @block %zu errno=%d (%s)\n",i,errno,strerror(errno));} break; } }
  printf("*** %s: registered %d/%zu blocks with ~38G free (occupier held) ***\n",mode.c_str(),okreg,pool.size());
  phase("P3_registered_under_pressure");

  // P4: free occupier
  if(occ){ (void)hipFree(occ); (void)hipDeviceSynchronize(); }
  phase("P4_freed_occupier");

  // cleanup
  for(size_t i=0;i<mrs.size();i++){ ibv_dereg_mr(mrs[i]); if(fds[i]>=0) close(fds[i]); } (void)hipDeviceSynchronize();
  for(auto p:pool) (void)hipFree(p); (void)hipDeviceSynchronize();
  printf("### done ###\n");
  return 0;
}
