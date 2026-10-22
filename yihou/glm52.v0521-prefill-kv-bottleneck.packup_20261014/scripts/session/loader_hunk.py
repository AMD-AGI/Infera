
        from sglang.srt.models.yihou_expertn import keep_n as expertn_keep_n
        from sglang.srt.models.yihou_expertn import weights_iterator as expertn_weights

        _yihou_n = (
            expertn_keep_n(source.model_config.hf_config)
            if source.model_config is not None else None
        )
        if _yihou_n:
            if not use_safetensors or self.load_config.draft_model_idx is not None or source.model_config.is_draft_model:
                raise ValueError("YIHOU_EXPERTN supports the safetensors target model only (no draft/MTP)")
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            return ((source.prefix + name, tensor) for name, tensor in expertn_weights(hf_weights_files, _yihou_n))
