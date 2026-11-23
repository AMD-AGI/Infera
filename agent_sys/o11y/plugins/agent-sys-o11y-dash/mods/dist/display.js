// protocol/schemas/common.v1.schema.json
var common_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "common.v1.schema.json",
  title: "Shared definitions of the o11y protocol (referenced by $ref)",
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
    },
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    },
    writer: {
      type: "object",
      required: [
        "session_id",
        "role",
        "source"
      ],
      properties: {
        session_id: {
          type: "string",
          minLength: 1
        },
        session_name: {
          type: "string"
        },
        role: {
          enum: [
            "main",
            "child",
            "analyzer",
            "other"
          ],
          description: "other = the session belongs to no fleet"
        },
        source: {
          enum: [
            "written",
            "extracted",
            "detected"
          ],
          description: "written = the session recorded its own fact; extracted = LLM-derived from history; detected = deterministic detector"
        },
        from: {
          type: "array",
          items: {
            type: "object",
            required: [
              "path"
            ],
            properties: {
              path: {
                type: "string",
                minLength: 1
              },
              line: {
                type: "integer",
                minimum: 1
              },
              offset: {
                type: "integer",
                minimum: 0
              },
              uuid: {
                type: "string"
              },
              pointer: {
                type: "string",
                pattern: "^(/.*)?$"
              }
            }
          }
        }
      }
    },
    facts_header: {
      type: "object",
      required: [
        "v",
        "generated_at",
        "stale_after_s",
        "sources"
      ],
      properties: {
        v: {
          const: 1
        },
        generated_at: {
          $ref: "#/$defs/time"
        },
        stale_after_s: {
          type: "number",
          exclusiveMinimum: 0
        },
        sources: {
          type: "array",
          items: {
            type: "string"
          }
        },
        problems: {
          type: "array",
          items: {
            type: "string"
          }
        }
      }
    }
  }
};

// protocol/schemas/config.trigger.v1.schema.json
var config_trigger_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "config.trigger.v1.schema.json",
  title: "Trigger subsystem config (written by display forms; defaults apply to missing keys)",
  type: "object",
  additionalProperties: false,
  required: [
    "v"
  ],
  properties: {
    v: {
      const: 1
    },
    updated_at: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    task_book: {
      type: "string",
      default: "mission.md",
      description: "path of the task book, relative to the main session cwd or absolute"
    },
    kick: {
      type: "object",
      additionalProperties: false,
      properties: {
        interval_s: {
          type: "number",
          default: 1800,
          minimum: 60
        },
        template: {
          type: "string",
          default: "Make sure you are actively pushing the work forward and not idle. Re-read the task book {task_book} and confirm the current work has not drifted from it. Record run starts/ends, plans and campaigns with the {ledger_tool} tool. Relay this check to the child sessions you manage: {children}."
        }
      }
    },
    packup: {
      type: "object",
      additionalProperties: false,
      properties: {
        interval_s: {
          type: "number",
          default: 3600,
          minimum: 60
        },
        every_n_kicks: {
          type: "integer",
          default: 3,
          minimum: 1,
          description: "packup replaces every n-th kick"
        },
        template: {
          type: "string",
          default: "Pack up the experiments run since the last packup according to the project conventions and make sure the run ledger is complete. Relay to child sessions: {children}."
        }
      }
    },
    refresh: {
      type: "object",
      additionalProperties: false,
      properties: {
        every_n_tool_calls: {
          type: "integer",
          default: 10,
          minimum: 1
        },
        every_n_turns: {
          type: "integer",
          default: 3,
          minimum: 1
        }
      }
    },
    log: {
      type: "object",
      additionalProperties: false,
      properties: {
        every_n_tool_calls: {
          type: "integer",
          default: 30,
          minimum: 1
        },
        every_n_turns: {
          type: "integer",
          default: 5,
          minimum: 1
        },
        slow_interval_s: {
          type: "number",
          default: 900,
          minimum: 60,
          description: "timer-driven log analysis even when idle"
        }
      }
    }
  }
};

// protocol/schemas/config.analysis.v1.schema.json
var config_analysis_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "config.analysis.v1.schema.json",
  title: "Analysis subsystem config",
  type: "object",
  additionalProperties: false,
  required: [
    "v"
  ],
  properties: {
    v: {
      const: 1
    },
    updated_at: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    log: {
      type: "object",
      additionalProperties: false,
      properties: {
        model: {
          type: "string",
          default: "haiku"
        },
        effort: {
          enum: [
            "low",
            "medium",
            "high",
            "xhigh",
            "max"
          ],
          default: "low"
        },
        prompt: {
          type: "string",
          default: "prompts/log.md",
          description: "prompt file (relative to the plugin) or inline text"
        },
        lookback_bytes: {
          type: "integer",
          default: 8e5,
          minimum: 0,
          description: "first sight of a history: the model reads only its last N bytes"
        }
      }
    },
    extract: {
      type: "object",
      additionalProperties: false,
      properties: {
        model: {
          type: "string",
          default: "haiku"
        },
        effort: {
          enum: [
            "low",
            "medium",
            "high",
            "xhigh",
            "max"
          ],
          default: "low"
        },
        prompt: {
          type: "string",
          default: "prompts/extract.md",
          description: "prompt file (relative to the plugin) or inline text"
        },
        lookback_bytes: {
          type: "integer",
          default: 2e5,
          minimum: 0,
          description: "first sight of a history: the model reads only its last N bytes"
        }
      }
    },
    sources: {
      type: "object",
      additionalProperties: false,
      properties: {
        machine_pool_dir: {
          type: [
            "string",
            "null"
          ],
          default: null,
          description: "machine-pool state dir (machine-pool skill); null = no machine-pool source"
        },
        e2e_flow_globs: {
          type: "array",
          items: {
            type: "string"
          },
          default: [],
          description: "where to look for e2e-flow kernel_table.json / integration_report.json"
        },
        otel_dir: {
          type: [
            "string",
            "null"
          ],
          default: null
        }
      }
    }
  }
};

// protocol/schemas/config.display.v1.schema.json
var config_display_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "config.display.v1.schema.json",
  title: "Display subsystem config",
  type: "object",
  additionalProperties: false,
  required: [
    "v"
  ],
  properties: {
    v: {
      const: 1
    },
    updated_at: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    poll_ms: {
      type: "integer",
      default: 1e3,
      minimum: 200
    },
    mode: {
      enum: [
        "narrow",
        "wide"
      ],
      default: "narrow"
    },
    recent_runs_n: {
      type: "integer",
      default: 3,
      minimum: 1
    },
    log_lines: {
      type: "integer",
      default: 50,
      minimum: 1
    }
  }
};

// protocol/schemas/facts.general.v1.schema.json
var facts_general_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "facts.general.v1.schema.json",
  title: "Display facts: general tab",
  type: "object",
  required: [
    "v",
    "task",
    "best",
    "spend",
    "progress"
  ],
  properties: {
    v: {
      const: 1
    },
    task: {
      type: [
        "object",
        "null"
      ],
      required: [
        "task_id",
        "title",
        "metric",
        "baseline"
      ],
      properties: {
        task_id: {
          type: "string"
        },
        title: {
          type: "string"
        },
        metric: {
          type: "object",
          required: [
            "name",
            "unit",
            "direction"
          ],
          properties: {
            name: {
              type: "string"
            },
            unit: {
              type: "string"
            },
            direction: {
              enum: [
                "higher",
                "lower"
              ]
            }
          }
        },
        baseline: {
          type: "object",
          required: [
            "value"
          ],
          properties: {
            value: {
              type: "number"
            },
            label: {
              type: "string"
            }
          }
        },
        target: {
          type: [
            "object",
            "null"
          ],
          required: [
            "value"
          ],
          properties: {
            value: {
              type: "number"
            },
            rule: {
              type: "string"
            }
          }
        },
        phase: {
          type: "string"
        }
      }
    },
    best: {
      type: [
        "object",
        "null"
      ],
      required: [
        "value",
        "run_ids"
      ],
      properties: {
        value: {
          type: "number"
        },
        run_ids: {
          type: "array",
          items: {
            type: "string"
          }
        },
        aggregate: {
          type: "string"
        },
        stack: {
          type: "string"
        },
        vs_baseline_pct: {
          type: [
            "number",
            "null"
          ]
        },
        of_target_pct: {
          type: [
            "number",
            "null"
          ]
        },
        at: {
          $ref: "common.v1.schema.json#/$defs/time"
        }
      }
    },
    spend: {
      type: "object",
      required: [
        "tokens_new",
        "tokens_cache_read",
        "wall_s"
      ],
      properties: {
        tokens_new: {
          type: "integer",
          minimum: 0
        },
        tokens_cache_read: {
          type: "integer",
          minimum: 0
        },
        steps_crossed: {
          type: "integer",
          minimum: 0
        },
        wall_s: {
          type: "number",
          minimum: 0
        },
        started_at: {
          $ref: "common.v1.schema.json#/$defs/time"
        },
        sessions: {
          type: "array",
          items: {
            type: "object",
            required: [
              "session_id",
              "tokens"
            ],
            properties: {
              session_id: {
                type: "string"
              },
              name: {
                type: [
                  "string",
                  "null"
                ]
              },
              role: {
                type: [
                  "string",
                  "null"
                ]
              },
              tokens: {
                type: "object",
                required: [
                  "input",
                  "cache_creation",
                  "cache_read",
                  "output",
                  "new"
                ],
                properties: {
                  input: {
                    type: "integer",
                    minimum: 0
                  },
                  cache_creation: {
                    type: "integer",
                    minimum: 0
                  },
                  cache_read: {
                    type: "integer",
                    minimum: 0
                  },
                  output: {
                    type: "integer",
                    minimum: 0
                  },
                  new: {
                    type: "integer",
                    minimum: 0,
                    description: "input + cache_creation + output: the milestone total (spec \xA76.2)"
                  }
                }
              },
              steps_crossed: {
                type: "integer",
                minimum: 0
              },
              last_at: {
                $ref: "common.v1.schema.json#/$defs/time"
              }
            }
          }
        }
      }
    },
    progress: {
      type: "object",
      required: [
        "gain_abs",
        "gain_pct"
      ],
      properties: {
        gain_abs: {
          type: [
            "number",
            "null"
          ]
        },
        gain_pct: {
          type: [
            "number",
            "null"
          ]
        },
        per_hour_abs: {
          type: [
            "number",
            "null"
          ]
        },
        per_hour_pct: {
          type: [
            "number",
            "null"
          ]
        },
        per_mtok_abs: {
          type: [
            "number",
            "null"
          ]
        },
        per_mtok_pct: {
          type: [
            "number",
            "null"
          ]
        }
      }
    }
  },
  allOf: [
    {
      $ref: "common.v1.schema.json#/$defs/facts_header"
    }
  ]
};

// protocol/schemas/facts.detail.v1.schema.json
var facts_detail_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "facts.detail.v1.schema.json",
  title: "Display facts: detail tab",
  type: "object",
  required: [
    "v",
    "current_runs",
    "recent_runs",
    "next",
    "machines"
  ],
  properties: {
    v: {
      const: 1
    },
    current_runs: {
      type: "array",
      items: {
        type: "object",
        required: [
          "run_id",
          "theme"
        ],
        properties: {
          run_id: {
            type: "string"
          },
          theme: {
            type: "string"
          },
          hypothesis: {
            type: [
              "string",
              "null"
            ]
          },
          point: {
            type: [
              "string",
              "null"
            ]
          },
          machines: {
            type: "array",
            items: {
              type: "string"
            }
          },
          started_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          session: {
            type: [
              "string",
              "null"
            ]
          }
        }
      }
    },
    recent_runs: {
      type: "array",
      items: {
        type: "object",
        required: [
          "run_id",
          "theme",
          "status"
        ],
        properties: {
          run_id: {
            type: "string"
          },
          theme: {
            type: "string"
          },
          point: {
            type: [
              "string",
              "null"
            ]
          },
          status: {
            type: "string"
          },
          value: {
            type: [
              "number",
              "null"
            ]
          },
          delta_pct: {
            type: [
              "number",
              "null"
            ]
          },
          delta_vs: {
            type: [
              "string",
              "null"
            ]
          },
          new_best: {
            type: "boolean"
          },
          started_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          ended_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          conclusion: {
            type: [
              "string",
              "null"
            ]
          }
        }
      }
    },
    next: {
      type: "array",
      items: {
        type: "object",
        required: [
          "theme"
        ],
        properties: {
          theme: {
            type: "string"
          },
          why: {
            type: [
              "string",
              "null"
            ]
          },
          point: {
            type: [
              "string",
              "null"
            ]
          },
          blocked_on: {
            type: [
              "string",
              "null"
            ]
          }
        }
      }
    },
    machines: {
      type: "array",
      items: {
        type: "object",
        required: [
          "node"
        ],
        properties: {
          node: {
            type: "string"
          },
          cluster: {
            type: "string"
          },
          jobid: {
            type: "string"
          },
          runs: {
            type: [
              "string",
              "null"
            ]
          },
          session: {
            type: [
              "string",
              "null"
            ]
          },
          since: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          until: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          gpu_util: {
            type: [
              "number",
              "null"
            ]
          },
          source: {
            enum: [
              "ledger",
              "pool",
              "both"
            ]
          }
        }
      }
    }
  },
  allOf: [
    {
      $ref: "common.v1.schema.json#/$defs/facts_header"
    }
  ]
};

// protocol/schemas/facts.kernelforge.v1.schema.json
var facts_kernelforge_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "facts.kernelforge.v1.schema.json",
  title: "Display facts: KernelForge tab",
  type: "object",
  required: [
    "v",
    "hot_ops",
    "campaigns",
    "ops"
  ],
  properties: {
    v: {
      const: 1
    },
    hot_ops: {
      type: "array",
      items: {
        type: "object",
        required: [
          "profile_id",
          "role",
          "ops"
        ],
        properties: {
          profile_id: {
            type: "string"
          },
          role: {
            type: "string"
          },
          scope: {
            type: [
              "string",
              "null"
            ]
          },
          ops: {
            type: "array",
            items: {
              type: "object",
              required: [
                "name",
                "share_pct"
              ],
              properties: {
                name: {
                  type: "string"
                },
                share_pct: {
                  type: "number"
                },
                rank: {
                  type: "integer"
                },
                op_id: {
                  type: [
                    "string",
                    "null"
                  ]
                }
              }
            }
          }
        }
      }
    },
    campaigns: {
      type: "array",
      items: {
        type: "object",
        required: [
          "campaign_root",
          "operator_ids",
          "status"
        ],
        properties: {
          campaign_root: {
            type: "string"
          },
          operator_ids: {
            type: "array",
            items: {
              type: "string"
            }
          },
          engine: {
            type: [
              "string",
              "null"
            ]
          },
          status: {
            enum: [
              "running",
              "paused",
              "completed",
              "interrupted",
              "stale",
              "lost",
              "unknown"
            ]
          },
          started_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          age_s: {
            type: [
              "number",
              "null"
            ]
          },
          active_s: {
            type: [
              "number",
              "null"
            ]
          },
          iterations: {
            type: [
              "integer",
              "null"
            ]
          },
          kept: {
            type: [
              "integer",
              "null"
            ]
          },
          start_speedup: {
            type: [
              "number",
              "null"
            ]
          },
          best_speedup: {
            type: [
              "number",
              "null"
            ]
          },
          gain_pct: {
            type: [
              "number",
              "null"
            ]
          },
          gain_pct_per_hour: {
            type: [
              "number",
              "null"
            ]
          },
          last_event_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          termination_reason: {
            type: [
              "string",
              "null"
            ]
          },
          machine: {
            type: [
              "string",
              "null"
            ]
          },
          problem: {
            type: [
              "string",
              "null"
            ]
          }
        }
      }
    },
    ops: {
      type: "array",
      items: {
        type: "object",
        required: [
          "op_id",
          "stage"
        ],
        properties: {
          op_id: {
            type: "string"
          },
          stage: {
            type: "string"
          },
          share_pct: {
            type: "object",
            additionalProperties: {
              type: "number"
            }
          },
          e2e_verdict: {
            type: [
              "string",
              "null"
            ]
          },
          e2e_delta_pct: {
            type: [
              "number",
              "null"
            ]
          },
          campaign_root: {
            type: [
              "string",
              "null"
            ]
          }
        }
      }
    }
  },
  allOf: [
    {
      $ref: "common.v1.schema.json#/$defs/facts_header"
    }
  ]
};

// protocol/schemas/registry.sessions.v1.schema.json
var registry_sessions_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "registry.sessions.v1.schema.json",
  title: "Normalised fleet registry (registry/sessions.json)",
  type: "object",
  required: [
    "v",
    "generated_at",
    "sessions"
  ],
  properties: {
    v: {
      const: 1
    },
    generated_at: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    main_session_id: {
      type: [
        "string",
        "null"
      ]
    },
    sessions: {
      type: "array",
      items: {
        type: "object",
        required: [
          "session_id",
          "pid",
          "alive",
          "role",
          "status"
        ],
        properties: {
          session_id: {
            type: "string",
            minLength: 1
          },
          pid: {
            type: "integer",
            minimum: 1
          },
          proc_start: {
            type: [
              "string",
              "null"
            ]
          },
          alive: {
            type: "boolean"
          },
          crashed: {
            type: "boolean",
            description: "record present, process gone or procStart mismatch; a dead former main stays listed (active false) for a retention window"
          },
          name: {
            type: [
              "string",
              "null"
            ]
          },
          cwd: {
            type: [
              "string",
              "null"
            ]
          },
          tmux: {
            type: [
              "string",
              "null"
            ]
          },
          transcript_path: {
            type: [
              "string",
              "null"
            ]
          },
          status: {
            enum: [
              "busy",
              "idle",
              "waiting",
              "shell",
              "unknown"
            ]
          },
          waiting_for: {
            type: [
              "string",
              "null"
            ]
          },
          status_since: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          role: {
            enum: [
              "main",
              "child",
              "other"
            ]
          },
          parent_session_id: {
            type: [
              "string",
              "null"
            ]
          },
          lineage: {
            type: "array",
            items: {
              type: "string"
            },
            description: "earlier session ids of the same process (/clear)"
          },
          started_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          updated_at: {
            $ref: "common.v1.schema.json#/$defs/time"
          },
          version: {
            type: [
              "string",
              "null"
            ]
          },
          kind: {
            type: [
              "string",
              "null"
            ]
          },
          purpose: {
            type: [
              "string",
              "null"
            ]
          },
          active: {
            type: "boolean",
            description: "true for the fleet's active main session (the display shows only for it)"
          },
          fleet_dir: {
            type: [
              "string",
              "null"
            ],
            description: "fleet root this session belongs to"
          },
          config_dir: {
            type: [
              "string",
              "null"
            ],
            description: "the session's Claude configuration directory"
          },
          launch_dir: {
            type: [
              "string",
              "null"
            ],
            description: "directory claude was started in (CLAUDE_PROJECT_DIR)"
          },
          record_stale: {
            type: "boolean",
            description: "true when Claude Code's record was torn or unparsable and this entry is the last good value (spec \xA73.4)"
          },
          messaging_socket: {
            type: [
              "string",
              "null"
            ],
            description: "Claude Code's messagingSocketPath (full path): a machine-pool notify `uds:<path>` belongs to this session only on an exact match"
          }
        }
      }
    }
  }
};

// protocol/schemas/trigger_fire.v1.schema.json
var trigger_fire_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "trigger_fire.v1.schema.json",
  title: "One trigger firing (triggers/<seq>.json)",
  type: "object",
  required: [
    "v",
    "id",
    "ts",
    "seq",
    "kind",
    "outcome",
    "writer"
  ],
  properties: {
    v: {
      const: 1
    },
    id: {
      $ref: "common.v1.schema.json#/$defs/id"
    },
    ts: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    seq: {
      type: "integer",
      minimum: 0
    },
    kind: {
      enum: [
        "kick",
        "packup",
        "question"
      ]
    },
    outcome: {
      enum: [
        "submitted",
        "queued",
        "skipped_pending",
        "refused",
        "dropped",
        "failed"
      ]
    },
    session_id: {
      type: "string"
    },
    detail: {
      type: "string"
    },
    counters: {
      type: "object",
      properties: {
        tool_calls: {
          type: "integer",
          minimum: 0
        },
        turns: {
          type: "integer",
          minimum: 0
        },
        kicks: {
          type: "integer",
          minimum: 0
        }
      }
    },
    writer: {
      $ref: "common.v1.schema.json#/$defs/writer"
    },
    text: {
      type: "string",
      description: "exact text handed to Claude Code"
    },
    subject_session_id: {
      type: "string",
      description: "question: the unregistered session"
    }
  }
};

// protocol/schemas/requests.v1.schema.json
var requests_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "requests.v1.schema.json",
  title: "Analysis requests (requests/analysis.json), whole file written by T",
  type: "object",
  required: [
    "v"
  ],
  additionalProperties: false,
  properties: {
    v: {
      const: 1
    },
    log: {
      type: "object",
      required: [
        "seq",
        "ts"
      ],
      additionalProperties: false,
      properties: {
        seq: {
          type: "integer",
          minimum: 0,
          description: "bumped by T on each request; A serves up to it"
        },
        ts: {
          $ref: "common.v1.schema.json#/$defs/time"
        },
        reason: {
          enum: [
            "tool_calls",
            "turns",
            "timer",
            "manual"
          ]
        }
      }
    },
    extract: {
      type: "object",
      required: [
        "seq",
        "ts"
      ],
      additionalProperties: false,
      properties: {
        seq: {
          type: "integer",
          minimum: 0,
          description: "bumped by T on each request; A serves up to it"
        },
        ts: {
          $ref: "common.v1.schema.json#/$defs/time"
        },
        reason: {
          enum: [
            "tool_calls",
            "turns",
            "timer",
            "manual"
          ]
        }
      }
    },
    facts: {
      type: "object",
      required: [
        "seq",
        "ts"
      ],
      additionalProperties: false,
      properties: {
        seq: {
          type: "integer",
          minimum: 0,
          description: "bumped by T on each request; A serves up to it"
        },
        ts: {
          $ref: "common.v1.schema.json#/$defs/time"
        },
        reason: {
          enum: [
            "tool_calls",
            "turns",
            "timer",
            "manual"
          ]
        }
      }
    }
  }
};

// protocol/schemas/otel_health.v1.schema.json
var otel_health_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "otel_health.v1.schema.json",
  title: "Collector health (otel/health.json), whole file written by T",
  type: "object",
  required: [
    "v",
    "ts",
    "healthy",
    "outages"
  ],
  additionalProperties: false,
  properties: {
    v: {
      const: 1
    },
    ts: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    healthy: {
      type: "boolean"
    },
    outages: {
      type: "integer",
      minimum: 0,
      description: "count of healthy\u2192down edges so far"
    },
    last_change: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    down_since: {
      $ref: "common.v1.schema.json#/$defs/time",
      description: "start of the current outage; absent while healthy"
    }
  }
};

// protocol/schemas/session_pin.v1.schema.json
var session_pin_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "session_pin.v1.schema.json",
  title: "Fleet-root pin of one session ($AGENT_SYS_HOME/state/o11y/sessions/<sid>.json), written by T",
  type: "object",
  required: [
    "v",
    "session_id",
    "fleet_dir",
    "pinned_at"
  ],
  additionalProperties: false,
  properties: {
    v: {
      const: 1
    },
    session_id: {
      type: "string",
      minLength: 1
    },
    fleet_dir: {
      type: "string",
      pattern: "^/"
    },
    launch_dir: {
      type: "string"
    },
    pinned_at: {
      $ref: "common.v1.schema.json#/$defs/time"
    },
    lineage: {
      type: "array",
      items: {
        type: "string"
      },
      description: "earlier session ids of the same process (/clear); the pin carries over"
    }
  }
};

// protocol/ts/index.ts
var SCHEMAS = {
  "config.trigger": config_trigger_v1_schema_default,
  "config.analysis": config_analysis_v1_schema_default,
  "config.display": config_display_v1_schema_default,
  "facts.general": facts_general_v1_schema_default,
  "facts.detail": facts_detail_v1_schema_default,
  "facts.kernelforge": facts_kernelforge_v1_schema_default,
  "registry.sessions": registry_sessions_v1_schema_default,
  trigger_fire: trigger_fire_v1_schema_default,
  requests: requests_v1_schema_default,
  otel_health: otel_health_v1_schema_default,
  session_pin: session_pin_v1_schema_default
};
var P = {
  config: (name) => `config/${name}.json`,
  facts: (tab2) => `facts/${tab2}.json`,
  sessions: "registry/sessions.json",
  launches: "registry/launches.jsonl",
  events: "logs/events.jsonl",
  triggers: "triggers",
  requests: "requests/analysis.json",
  otelHealth: "otel/health.json",
  triggerState: "state/trigger.json"
};
var trim = (p) => p.length > 1 ? p.replace(/\/+$/, "") : p;
async function slug(dir) {
  const name = dir.replace(/[^A-Za-z0-9]/gu, "-");
  if (name.length <= 200) return name;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(dir)));
  return `${name.slice(0, 200)}-${Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 12)}`;
}
function prefixOf(env) {
  const h = env.AGENT_SYS_HOME;
  return h && h.startsWith("/") ? trim(h) : `${trim(env.HOME || "/tmp")}/.infera_agent_sys`;
}
var pinPath = (env, sid) => `${prefixOf(env)}/state/o11y/sessions/${sid}.json`;
async function derivedFleetDir(env, launchDir) {
  const f = env.AGENT_SYS_O11Y_FLEET_DIR;
  if (f && f.startsWith("/")) return f;
  return `${prefixOf(env)}/state/o11y/fleets/${await slug(launchDir)}`;
}
async function readEnv($) {
  return {
    AGENT_SYS_O11Y_FLEET_DIR: await $.env.get("AGENT_SYS_O11Y_FLEET_DIR"),
    AGENT_SYS_HOME: await $.env.get("AGENT_SYS_HOME"),
    HOME: await $.env.get("HOME"),
    CLAUDE_CONFIG_DIR: await $.env.get("CLAUDE_CONFIG_DIR")
  };
}
async function fleetDirOf($, env, sid) {
  const pin = (await readDoc($, pinPath(env, sid), "session_pin")).value;
  if (pin) return { fleet: pin.fleet_dir, pinned: pin };
  return { fleet: await derivedFleetDir(env, String(await $.session.root())), pinned: null };
}
var nowIso = (ms) => new Date(ms ?? Date.now()).toISOString();
function typeOf(x) {
  if (x === null) return "null";
  if (Array.isArray(x)) return "array";
  if (typeof x === "number") return Number.isInteger(x) ? "integer" : "number";
  return typeof x;
}
function deref(node, root3) {
  while (node && typeof node.$ref === "string") {
    const [file, frag] = node.$ref.split("#");
    if (file) root3 = file === "common.v1.schema.json" ? common_v1_schema_default : null;
    if (!root3 || !frag?.startsWith("/$defs/")) throw new Error(`unresolvable $ref ${node.$ref}`);
    node = root3.$defs[frag.slice(7)];
  }
  return [node, root3];
}
function validate(schema, value, root3 = schema, at = "") {
  const [s, r] = deref(schema, root3);
  if (!s || typeof s !== "object") return [];
  const where = at || "/";
  const out = [];
  for (const sub of s.allOf ?? []) out.push(...validate(sub, value, r, at));
  const t = typeOf(value);
  if (s.type) {
    const types = Array.isArray(s.type) ? s.type : [s.type];
    if (!types.some((ty) => ty === t || ty === "number" && t === "integer")) return [...out, `${where}: expected ${types.join("|")}, got ${t}`];
  }
  if (s.const !== void 0 && value !== s.const) out.push(`${where}: must be ${JSON.stringify(s.const)}`);
  if (s.enum && !s.enum.includes(value)) out.push(`${where}: must be one of ${s.enum.join(", ")}`);
  if (typeof value === "number") {
    if (typeof s.minimum === "number" && value < s.minimum) out.push(`${where}: must be >= ${s.minimum}`);
    if (typeof s.exclusiveMinimum === "number" && value <= s.exclusiveMinimum) out.push(`${where}: must be > ${s.exclusiveMinimum}`);
  }
  if (typeof value === "string") {
    const n = [...value].length;
    if (typeof s.minLength === "number" && n < s.minLength) out.push(`${where}: shorter than ${s.minLength}`);
    if (typeof s.maxLength === "number" && n > s.maxLength) out.push(`${where}: longer than ${s.maxLength}`);
    if (typeof s.pattern === "string" && !new RegExp(s.pattern, "u").test(value)) out.push(`${where}: does not match ${s.pattern}`);
  }
  if (t === "object") {
    const obj = value;
    for (const k of s.required ?? []) if (!(k in obj)) out.push(`${at}/${k}: required`);
    for (const [k, v] of Object.entries(obj)) {
      if (s.properties?.[k]) out.push(...validate(s.properties[k], v, r, `${at}/${k}`));
      else if (s.additionalProperties === false) out.push(`${at}/${k}: unknown key${didYouMean(k, Object.keys(s.properties ?? {}))}`);
    }
  }
  if (t === "array") {
    const arr = value;
    if (typeof s.minItems === "number" && arr.length < s.minItems) out.push(`${where}: needs ${s.minItems}+ items`);
    if (s.items) arr.forEach((v, i) => out.push(...validate(s.items, v, r, `${at}/${i}`)));
  }
  return out;
}
function distance(a, b) {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) row = row.reduce((n, v, j) => [...n, j === 0 ? i : Math.min(v + 1, n[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))], []);
  return row[b.length];
}
function didYouMean(key, known) {
  const best = known.map((k) => [k, distance(key, k)]).sort((x, y) => x[1] - y[1])[0];
  return best && best[1] <= Math.max(2, key.length / 3) ? ` (did you mean "${best[0]}"?)` : "";
}
function withDefaults(schema, value, root3 = schema) {
  const [s, r] = deref(schema, root3);
  if (!s || s.type !== "object" || !s.properties) return value;
  const out = { ...value ?? {} };
  for (const [k, ps0] of Object.entries(s.properties)) {
    const [ps] = deref(ps0, r);
    if (out[k] === void 0 && ps.default !== void 0) out[k] = JSON.parse(JSON.stringify(ps.default));
    if (ps.type === "object" && ps.properties) out[k] = withDefaults(ps, out[k], r);
  }
  return out;
}
async function readDoc($, path, schemaName) {
  try {
    if (!await $.fs.exists(path)) return { value: null, problem: null, mtimeMs: 0 };
    const st = await $.fs.stat(path);
    const value = JSON.parse(await $.fs.read(path));
    const probs = schemaName ? validate(SCHEMAS[schemaName], value) : [];
    return probs.length ? { value: null, problem: `${path}: ${probs[0]}`, mtimeMs: st.mtimeMs } : { value, problem: null, mtimeMs: st.mtimeMs };
  } catch (err) {
    return { value: null, problem: `${path}: ${err.message}`, mtimeMs: 0 };
  }
}
async function loadConfig($, path, name, prev) {
  const base = prev ?? { cfg: withDefaults(SCHEMAS[name], { v: 1 }), mtimeMs: -1, reported: null, report: null };
  const r = await readDoc($, path, name);
  if (r.mtimeMs === base.mtimeMs) return { ...base, report: null };
  if (r.problem) return { cfg: base.cfg, mtimeMs: r.mtimeMs, reported: r.problem, report: base.reported === r.problem ? null : r.problem };
  return { cfg: withDefaults(SCHEMAS[name], r.value ?? { v: 1 }), mtimeMs: r.mtimeMs, reported: null, report: null };
}
async function writeDoc($, path, value) {
  const tmp = `${path}.tmp.${Math.random().toString(36).slice(2, 8)}`;
  await $.fs.write(tmp, JSON.stringify(value, null, 1) + "\n");
  const r = await $.process.run(["mv", "-f", "--", tmp, path], { timeoutMs: 1e4 });
  if (r.exitCode !== 0) throw new Error(`mv ${tmp}: ${r.stderr.trim()}`);
}
async function readJsonl($, path, maxBytes = 1 << 20) {
  if (!await $.fs.exists(path)) return [];
  const st = await $.fs.stat(path);
  const offset = Math.max(0, st.size - maxBytes);
  const text2 = offset > 0 ? await $.fs.read(path, { offset, length: st.size - offset }) : await $.fs.read(path);
  const lines = text2.split("\n").slice(offset > 0 ? 1 : 0, -1);
  return lines.flatMap((l) => {
    try {
      return l ? [JSON.parse(l)] : [];
    } catch {
      return [];
    }
  });
}

// plugins/agent-sys-o11y-dash/mods/src/display/forms.ts
var FORM = "o11y-cfg";
var root = "";
var target = "trigger";
var draft = { v: 1 };
var note = "";
function fieldsOf(schema, at = []) {
  const out = [];
  for (const [k, ps] of Object.entries(schema.properties ?? {})) {
    if (k === "v" || k === "updated_at") continue;
    if (ps.type === "object" && ps.properties) out.push(...fieldsOf(ps, [...at, k]));
    else out.push({ path: [...at, k], schema: ps });
  }
  return out;
}
var getAt = (obj, path) => path.reduce((o, k) => o == null ? void 0 : o[k], obj);
function setAt(obj, path, value) {
  let o = obj;
  for (const k of path.slice(0, -1)) o = o[k] ??= {};
  o[path[path.length - 1]] = value;
}
async function openForm($, fleetRoot2, t) {
  root = fleetRoot2;
  target = t;
  const r = await readDoc($, `${root}/${P.config(t)}`);
  draft = r.value && typeof r.value === "object" ? r.value : { v: 1 };
  note = r.problem ? `current file unreadable: ${r.problem}` : `config/${t}.json`;
  await $.ui.open({ id: FORM, title: `o11y config: ${t}`, focus: true, closeOnEscape: true, rows: 30, columns: 90 });
}
function parse(f, raw) {
  const t = Array.isArray(f.schema.type) ? f.schema.type[0] : f.schema.type;
  if (t !== "number" && t !== "integer") return { value: raw };
  const n = Number(raw);
  if (raw.trim() === "" || !isFinite(n) || t === "integer" && !Number.isInteger(n)) return { problem: `${f.path.join(".")}: not a ${t}` };
  return { value: n };
}
async function save($, f, value) {
  const next = JSON.parse(JSON.stringify(draft));
  setAt(next, f.path, value);
  next.v = 1;
  next.updated_at = nowIso();
  const probs = validate(SCHEMAS[`config.${target}`], next);
  if (probs.length) note = `not saved: ${probs[0]}`;
  else {
    await writeDoc($, `${root}/${P.config(target)}`, next);
    draft = next;
    note = `saved ${f.path.join(".")} = ${JSON.stringify(value).slice(0, 60)}`;
  }
  $.ui.invalidate("ui.render");
}
async function renderForm($, e) {
  const { Box, Text, Input, Select } = $.ui.resolve(e);
  const schema = SCHEMAS[`config.${target}`];
  const full = withDefaults(schema, draft);
  const rows = fieldsOf(schema).map((f, i) => {
    const key = `cfg-${target}-${f.path.join(".")}`;
    const label = f.path.join(".");
    const cur = getAt(full, f.path);
    const isBool = f.schema.type === "boolean";
    const choices = isBool ? [true, false] : Array.isArray(f.schema.enum) ? f.schema.enum.filter((x) => x !== null) : null;
    if (choices) {
      return Select({
        key,
        label,
        value: String(cur),
        ...i === 0 ? { autoFocus: true } : {},
        options: choices.map((c) => ({ value: String(c), label: String(c) })),
        onSelect: (v) => save($, f, isBool ? v === "true" : v)
      });
    }
    return Input({
      key,
      label,
      value: String(cur ?? ""),
      submitLabel: "save",
      ...i === 0 ? { autoFocus: true } : {},
      onSubmit: (raw) => {
        const p = parse(f, raw);
        if (p.problem) {
          note = `not saved: ${p.problem}`;
          $.ui.invalidate("ui.render");
          return;
        }
        return save($, f, p.value);
      }
    });
  });
  return Box({ flexDirection: "column", children: [Text({ dimColor: true, wrap: "truncate-end", children: [`${note} \xB7 Tab/Enter to edit, Esc to close`] }), ...rows] });
}

// plugins/agent-sys-o11y-dash/mods/src/display/fmt.ts
function num(x, digits = 0) {
  return typeof x === "number" && isFinite(x) ? x.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "-";
}
var pct = (x) => typeof x === "number" && isFinite(x) ? `${x >= 0 ? "+" : ""}${x.toFixed(1)}%` : "-";
var tok = (n) => typeof n !== "number" ? "-" : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n);
function dur(s) {
  if (typeof s !== "number" || !isFinite(s)) return "-";
  const h = Math.floor(s / 3600);
  const m = Math.floor(s % 3600 / 60);
  return h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m`;
}
function ago(iso, now = Date.now()) {
  const t = Date.parse(iso);
  return isNaN(t) ? "-" : dur(Math.max(0, (now - t) / 1e3));
}

// plugins/agent-sys-o11y-dash/mods/src/display/general.ts
function general(g, wide, sessions2) {
  const out = [];
  const t = g.task;
  if (t) {
    if (wide) out.push([`${t.title}${t.phase ? ` [${t.phase}]` : ""}`]);
    out.push([`${t.metric.name} (${t.metric.unit}, ${t.metric.direction} is better)`]);
    out.push([`base ${num(t.baseline.value)}${wide && t.baseline.label ? ` (${t.baseline.label})` : ""} \xB7 target ${t.target ? num(t.target.value) : "-"}`]);
  } else out.push(["task: not recorded yet (ledger kind=task)", "yellow"]);
  const b = g.best;
  if (b) {
    out.push([`best ${num(b.value)}  ${pct(b.vs_baseline_pct)} vs base \xB7 ${b.of_target_pct == null ? "-" : b.of_target_pct.toFixed(1) + "%"} of target`, "green"]);
    if (wide) out.push([`  runs ${b.run_ids.join(", ")}${b.stack ? ` \xB7 ${b.stack}` : ""}`]);
  }
  const s = g.spend;
  out.push([`tokens ${tok(s.tokens_new)} new + ${tok(s.tokens_cache_read)} cache reads \xB7 ${dur(s.wall_s)}`]);
  const p = g.progress;
  out.push([`progress ${pct(p.gain_pct)} \xB7 ${pct(p.per_hour_pct)}/h \xB7 ${pct(p.per_mtok_pct)}/Mtok`]);
  if (!wide) return out;
  out.push([`  abs ${num(p.gain_abs, 1)} \xB7 ${num(p.per_hour_abs, 1)}/h \xB7 ${num(p.per_mtok_abs, 1)}/Mtok`]);
  out.push(["sessions:"]);
  for (const x of sessions2) {
    const u = (s.sessions ?? []).find((y) => y.session_id === x.session_id)?.tokens;
    const state = x.alive ? `${x.status}${x.waiting_for ? ` (${x.waiting_for})` : ""}` : "DEAD";
    out.push([`  ${x.active ? "*" : " "} ${x.role} ${x.name ?? x.session_id.slice(0, 8)} ${state} \xB7 ${tok(u?.new)} new ${tok(u?.cache_read)} cached`, x.alive ? void 0 : "red"]);
  }
  for (const pr of g.problems ?? []) out.push([`! ${pr}`, "yellow"]);
  return out;
}

// plugins/agent-sys-o11y-dash/mods/src/display/detail.ts
function detail(d, wide, recentN) {
  const out = [];
  for (const r of d.current_runs) {
    const m = (r.machines ?? []).map((x) => typeof x === "string" ? x : x.node).join(",");
    out.push([`\u25B6 ${r.run_id} ${r.theme}${r.point ? ` [${r.point}]` : ""}${m ? ` @${m}` : ""}`, "cyan"]);
    if (wide && r.hypothesis) out.push([`    hyp: ${r.hypothesis}`]);
  }
  if (!d.current_runs.length) out.push(["no run in progress", "dim"]);
  out.push([`last ${Math.min(recentN, d.recent_runs.length)} runs:`]);
  for (const r of d.recent_runs.slice(0, recentN)) {
    const c = r.new_best ? "green" : r.status === "failed" || r.status === "rejected" ? "red" : void 0;
    out.push([`  ${r.status.padEnd(8)} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, c]);
    if (wide && r.conclusion) out.push([`      ${r.conclusion}`]);
  }
  out.push([`next: ${d.next.length ? d.next[0].theme : "-"}`]);
  if (wide) for (const n of d.next.slice(1)) out.push([`      ${n.theme}${n.blocked_on ? ` (blocked: ${n.blocked_on})` : ""}`]);
  out.push([`machines (${d.machines.length}):`]);
  for (const m of wide ? d.machines : d.machines.slice(0, 4)) {
    out.push([`  ${m.node}${wide && m.cluster ? `/${m.cluster}` : ""} \u2192 ${m.runs ?? "idle"}${wide && m.gpu_util != null ? ` \xB7 gpu ${m.gpu_util}%` : ""}`]);
  }
  if (wide) for (const pr of d.problems ?? []) out.push([`! ${pr}`, "yellow"]);
  return out;
}

// plugins/agent-sys-o11y-dash/mods/src/display/kernelforge.ts
function kernelforge(k, wide) {
  const out = [];
  for (const h of k.hot_ops.slice(0, wide ? 3 : 1)) {
    out.push([`hot ops (${h.role}${wide && h.scope ? `, ${h.scope}` : ""}):`]);
    for (const o of h.ops.slice(0, wide ? 12 : 5)) out.push([`  ${o.share_pct.toFixed(1).padStart(5)}%  ${o.name}`]);
  }
  out.push([`campaigns (${k.campaigns.length}):`]);
  for (const c of k.campaigns) {
    const col = c.status === "running" ? "cyan" : c.status === "stale" || c.problem ? "yellow" : void 0;
    out.push([`  ${c.operator_ids.join(",")} ${c.status} ${pct(c.gain_pct)} (${pct(c.gain_pct_per_hour)}/h) ${dur(c.age_s)}`, col]);
    if (wide) out.push([`      ${c.campaign_root}${c.iterations != null ? ` \xB7 it ${c.iterations} kept ${c.kept ?? "-"}` : ""}${c.problem ? ` \xB7 ${c.problem}` : ""}`]);
  }
  out.push(["ops:"]);
  for (const o of wide ? k.ops : k.ops.slice(0, 6)) {
    out.push([`  ${o.op_id} ${o.stage}${o.e2e_verdict ? ` \xB7 e2e ${o.e2e_verdict} ${pct(o.e2e_delta_pct)}` : ""}`, o.e2e_verdict ? "green" : void 0]);
  }
  if (wide) for (const pr of k.problems ?? []) out.push([`! ${pr}`, "yellow"]);
  return out;
}

// plugins/agent-sys-o11y-dash/mods/src/display/log.ts
var LEVELS = ["INFO", "WARN", "ERROR"];
var NUM = { INFO: 9, WARN: 13, ERROR: 17 };
var COLOR = { WARN: "yellow", ERROR: "red" };
function log(events2, wide, n, minLevel2) {
  const min = NUM[minLevel2] ?? 9;
  const ev = events2.filter((x) => (x.severity_number ?? NUM[x.severity_text] ?? 9) >= min).slice(-n);
  const head = [`level \u2265 ${minLevel2} (f: change)`, "dim"];
  if (!ev.length) return [head, ["no events yet", "dim"]];
  return [
    head,
    ...ev.map((x) => {
      const who = wide && x.attributes?.["session.name"] ? ` ${x.attributes["session.name"]}` : "";
      return [`${String(x.ts ?? "").slice(11, 19)} ${String(x.severity_text).padEnd(5)}${wide ? ` ${x.event_name}` : ""}${who} ${x.body}`, COLOR[x.severity_text]];
    })
  ];
}

// plugins/agent-sys-o11y-dash/mods/src/display/frame.ts
var PANE = "o11y";
var MENU = "o11y-menu";
var TABS = ["general", "detail", "kernelforge", "log"];
var FACTS = ["general", "detail", "kernelforge"];
var SIZE = { narrow: { columns: 46, rows: 16 }, wide: { columns: 96, rows: 40 } };
var root2 = null;
var sessionId = "";
var active = false;
var registryKey = "";
var cfgState = null;
var tab = "general";
var mode = "narrow";
var bandOn = true;
var minLevel = "INFO";
var docs = {};
var sessions = [];
var events = [];
var eventsSize = -1;
var lastFire = null;
var lastFireName = "";
var polls = 0;
var poller = null;
var pollMs = 0;
var abs = (rel) => `${root2}/${rel}`;
var cfg = () => cfgState.cfg;
var isActive = () => active;
var fleetRoot = () => root2;
async function start($) {
  sessionId = await $.session.id();
  root2 = (await fleetDirOf($, await readEnv($), sessionId)).fleet;
  cfgState = await loadConfig($, abs(P.config("display")), "config.display", null);
  startPoller($);
}
function startPoller($) {
  if (poller) poller.cancel();
  pollMs = cfg().poll_ms;
  poller = $.clock.every(pollMs, () => poll($));
}
async function checkActive($) {
  const sid = await $.session.id() || sessionId;
  if (sid !== sessionId || !active && polls % 5 === 0) {
    sessionId = sid;
    root2 = (await fleetDirOf($, await readEnv($), sessionId)).fleet;
  }
  const sp = abs(P.sessions);
  const m = await $.fs.exists(sp) ? (await $.fs.stat(sp)).mtimeMs : 0;
  const key = `${root2}:${m}:${sessionId}`;
  if (key === registryKey) return;
  registryKey = key;
  sessions = (await readDoc($, sp, "registry.sessions")).value?.sessions ?? [];
  const mine = sessions.some((x) => x.session_id === sessionId && x.role === "main" && x.alive && x.active === true);
  if (mine && !active) {
    active = true;
    await poll($);
    mode = cfg().mode;
    await openPane($);
  } else if (!mine && active) {
    active = false;
    await $.ui.close({ id: PANE });
  }
  $.ui.invalidate("ui.render");
}
async function poll($) {
  if (!root2) return;
  polls++;
  let changed = false;
  try {
    await checkActive($);
    if (!active) return;
    const before = cfgState.mtimeMs;
    cfgState = await loadConfig($, abs(P.config("display")), "config.display", cfgState);
    if (cfgState.report) $.ui.log(`o11y: config/display.json not applied, the previous config stays: ${cfgState.report}`);
    changed ||= cfgState.mtimeMs !== before;
    for (const f of FACTS) {
      const path = abs(P.facts(f));
      const m = await $.fs.exists(path) ? (await $.fs.stat(path)).mtimeMs : 0;
      if (docs[f] && docs[f].mtimeMs === m) continue;
      const r = await readDoc($, path, `facts.${f}`);
      docs[f] = r.problem ? { ...r, problem: "invalid " + r.problem.replace(root2 + "/", "") } : r;
      changed = true;
    }
    const ep = abs(P.events);
    const es = await $.fs.exists(ep) ? (await $.fs.stat(ep)).size : 0;
    if (es !== eventsSize) {
      eventsSize = es;
      events = (await readJsonl($, ep, 256 * 1024)).filter((x) => x && x.v === 1 && typeof x.severity_text === "string");
      changed = true;
    }
    if (polls % 5 === 0 && await $.fs.exists(abs(P.triggers))) {
      const newest = (await $.fs.list(abs(P.triggers))).map((x) => x.name).filter((n) => /^\d+\.json$/.test(n)).sort().pop() ?? "";
      if (newest && newest !== lastFireName) {
        lastFireName = newest;
        lastFire = (await readDoc($, abs(`${P.triggers}/${newest}`), "trigger_fire")).value;
        changed = true;
      }
    }
  } catch (err) {
    $.ui.log(`o11y: display poll: ${err.message}`, { to: "debug" });
  }
  if (cfg().poll_ms !== pollMs) startPoller($);
  if (changed) $.ui.invalidate("ui.render");
}
function staleOf(name) {
  const d = docs[name]?.value;
  if (!d) return docs[name]?.problem ? "invalid" : "no data";
  const age = (Date.now() - Date.parse(d.generated_at)) / 1e3;
  return age > d.stale_after_s ? `stale ${dur(age)}` : null;
}
function linesOf(t, wide) {
  if (t === "log") return log(events, wide, cfg().log_lines, minLevel);
  const d = docs[t];
  if (!d?.value) return [[d?.problem ?? `no facts/${t}.json yet`, "yellow"]];
  if (t === "general") return general(d.value, wide, sessions);
  if (t === "detail") return detail(d.value, wide, cfg().recent_runs_n);
  return kernelforge(d.value, wide);
}
async function openPane($) {
  await $.ui.open({ id: PANE, title: "o11y", ...SIZE[mode] });
}
async function showTab($, t) {
  tab = t;
  await openPane($);
  $.ui.invalidate("ui.render");
}
async function setMode($, m) {
  mode = m === "narrow" || m === "wide" ? m : mode === "wide" ? "narrow" : "wide";
  await openPane($);
  $.ui.invalidate("ui.render");
}
function setBand($, arg) {
  bandOn = arg === "on" ? true : arg === "off" ? false : !bandOn;
  $.ui.invalidate("ui.render");
}
function text(Text, [s, color], key) {
  return Text({ key, wrap: "truncate-end", ...color === "dim" ? { dimColor: true } : color ? { color } : {}, children: [s] });
}
async function renderPane($, e) {
  const { Box, Text, Button } = $.ui.resolve(e);
  const wide = mode === "wide";
  const btn = (key, label, hotkey, dim, onPress) => Button({ key, label, hotkey, plain: true, dimColor: dim, onPress });
  const row = Box({
    flexDirection: "row",
    columnGap: 2,
    children: [
      ...TABS.map((t, i) => btn(`tab-${t}`, t === "kernelforge" && !wide ? "kf" : t, String(i + 1), t !== tab, () => showTab($, t))),
      btn("tab-mode", wide ? "narrow" : "wide", "m", true, () => setMode($)),
      ...tab === "log" ? [btn("tab-level", "level", "f", true, () => cycleLevel($))] : []
    ]
  });
  const stale = tab === "log" ? null : staleOf(tab);
  const meta = tab === "log" ? null : stale ? [`\u26A0 ${stale}`, "yellow"] : [`updated ${ago(docs[tab].value.generated_at)} ago`, "dim"];
  const body = linesOf(tab, wide).map((l, i) => text(Text, l, `l${i}`));
  return Box({ flexDirection: "column", children: [row, ...meta ? [text(Text, meta, "meta")] : [], ...body] });
}
function cycleLevel($) {
  minLevel = LEVELS[(LEVELS.indexOf(minLevel) + 1) % LEVELS.length];
  $.ui.invalidate("ui.render");
}
function renderBand($, e) {
  const { Text } = $.ui.resolve(e);
  const g = docs.general?.value;
  const parts = ["o11y"];
  if (g?.task && g.best) parts.push(`${g.task.metric.name} ${num(g.best.value)} ${pct(g.best.vs_baseline_pct)} (tgt ${g.task.target ? num(g.task.target.value) : "-"})`);
  else if (g?.task) parts.push(`${g.task.metric.name} base ${num(g.task.baseline.value)}`);
  if (g) parts.push(`tok ${tok(g.spend.tokens_new)}+${tok(g.spend.tokens_cache_read)}c`);
  const d = docs.detail?.value;
  if (d?.current_runs?.length) parts.push(`\u25B6 ${d.current_runs[0].run_id}`);
  const bad = [...events].reverse().find((x) => x.severity_text === "WARN" || x.severity_text === "ERROR");
  if (bad) parts.push(`${bad.severity_text} ${ago(bad.ts)} ago: ${bad.body}`);
  if (lastFire) parts.push(`${lastFire.kind} ${ago(lastFire.ts)} ago`);
  const st = staleOf("general");
  if (st) parts.push(`(${st})`);
  return Text({ wrap: "truncate-end", ...bad?.severity_text === "ERROR" ? { color: "red" } : {}, children: [parts.join(" \xB7 ")] });
}
var bandShown = () => active && bandOn;
async function openMenu($) {
  await $.ui.open({ id: MENU, title: "o11y menu", focus: true, closeOnEscape: true, holdToasts: true, rows: 4 });
}
async function renderMenu($, e) {
  const { Box, Text, Button } = $.ui.resolve(e);
  const items = [
    ["g", "general"],
    ["d", "detail"],
    ["k", "kernelforge"],
    ["l", "log"],
    ["m", mode === "wide" ? "narrow" : "wide"],
    ["b", bandOn ? "band off" : "band on"],
    ["t", "cfg trigger"],
    ["a", "cfg analysis"],
    ["v", "cfg display"],
    ["x", "close o11y"]
  ];
  const b = ([k, label]) => Button({ key: `menu-${k}`, label, hotkey: k, plain: true, onPress: () => menuPick($, k) });
  return Box({
    flexDirection: "column",
    children: [
      Text({ dimColor: true, children: ["o11y: press a key, Esc to cancel"] }),
      Box({ flexDirection: "row", columnGap: 2, children: items.slice(0, 5).map(b) }),
      Box({ flexDirection: "row", columnGap: 2, children: items.slice(5).map(b) })
    ]
  });
}
async function menuPick($, k) {
  await $.ui.close({ id: MENU });
  const tabs = { g: "general", d: "detail", k: "kernelforge", l: "log" };
  const forms = { t: "trigger", a: "analysis", v: "display" };
  if (tabs[k]) return showTab($, tabs[k]);
  if (forms[k]) return openForm($, root2, forms[k]);
  if (k === "m") return setMode($);
  if (k === "b") return setBand($);
  if (k === "x") await $.ui.close({ id: PANE });
}

// plugins/agent-sys-o11y-dash/mods/src/display/index.ts
var COMMANDS = [
  { name: "o11y", description: "o11y: menu (g/d/k/l tabs, m mode, b band, t/a/v config)" },
  { name: "o11y-general", description: "o11y: general tab" },
  { name: "o11y-detail", description: "o11y: detail tab (runs, plan, machines)" },
  { name: "o11y-kf", description: "o11y: KernelForge tab" },
  { name: "o11y-log", description: "o11y: log tab" },
  { name: "o11y-mode", description: "o11y: switch NARROW / WIDE", argumentHint: "[narrow|wide]" },
  { name: "o11y-band", description: "o11y: band above the prompt on/off", argumentHint: "[on|off]" },
  { name: "o11y-cfg-trigger", description: "o11y: edit the trigger configuration" },
  { name: "o11y-cfg-analysis", description: "o11y: edit the analysis configuration" },
  { name: "o11y-cfg-display", description: "o11y: edit the display configuration" }
];
var vetoReason = "";
async function command($, name, args) {
  if (!isActive()) {
    const why = vetoReason || `this session is not the active main of fleet ${fleetRoot()} (AGENT_SYS_O11Y=true or a first agent_sys skill use makes it one; the display follows within one trigger tick)`;
    return { text: `o11y display inactive: ${why}`, isTextDisplayOnly: true };
  }
  const a = (args ?? "").trim();
  if (name === "o11y") await openMenu($);
  else if (name === "o11y-general") await showTab($, "general");
  else if (name === "o11y-detail") await showTab($, "detail");
  else if (name === "o11y-kf") await showTab($, "kernelforge");
  else if (name === "o11y-log") await showTab($, "log");
  else if (name === "o11y-mode") await setMode($, a);
  else if (name === "o11y-band") setBand($, a);
  else if (name.startsWith("o11y-cfg-")) await openForm($, fleetRoot(), name.slice(9));
  return {};
}
function register(on) {
  on("session.start", async ($, e, next) => {
    for (const c of COMMANDS) {
      try {
        await $.command.register({ ...c, immediate: true });
      } catch (err) {
        $.ui.log(`o11y: /${c.name} not registered: ${err.message}`);
      }
    }
    const flag = await $.env.get("AGENT_SYS_O11Y");
    if (flag !== void 0 && flag !== "true") vetoReason = `AGENT_SYS_O11Y=${flag}; the display runs only in a fleet's main session`;
    else await start($);
    return next(e);
  });
  for (const c of COMMANDS) {
    on("command.run", { command: c.name }, async ($, e) => command($, e.command, e.args));
  }
  on("ui.render", { component: "Pane" }, async ($, e, next) => {
    if (e.requestId === FORM) return renderForm($, e);
    if (e.requestId === MENU) return renderMenu($, e);
    if (e.requestId === PANE && isActive()) return renderPane($, e);
    return next(e);
  });
  on("ui.render", { component: "AbovePrompt" }, async ($, e, next) => bandShown() ? renderBand($, e) : next(e));
}
export {
  register
};
