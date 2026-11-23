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
          default: 8e5,
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
  facts: (tab) => `facts/${tab}.json`,
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
function prefixOf(env2) {
  const h = env2.AGENT_SYS_HOME;
  return h && h.startsWith("/") ? trim(h) : `${trim(env2.HOME || "/tmp")}/.infera_agent_sys`;
}
var pinPath = (env2, sid) => `${prefixOf(env2)}/state/o11y/sessions/${sid}.json`;
function configDirOf(env2) {
  return env2.CLAUDE_CONFIG_DIR ? trim(env2.CLAUDE_CONFIG_DIR) : `${trim(env2.HOME || "/tmp")}/.claude`;
}
async function derivedFleetDir(env2, launchDir) {
  const f = env2.AGENT_SYS_O11Y_FLEET_DIR;
  if (f && f.startsWith("/")) return f;
  return `${prefixOf(env2)}/state/o11y/fleets/${await slug(launchDir)}`;
}
async function readEnv($) {
  return {
    AGENT_SYS_O11Y_FLEET_DIR: await $.env.get("AGENT_SYS_O11Y_FLEET_DIR"),
    AGENT_SYS_HOME: await $.env.get("AGENT_SYS_HOME"),
    HOME: await $.env.get("HOME"),
    CLAUDE_CONFIG_DIR: await $.env.get("CLAUDE_CONFIG_DIR")
  };
}
var nowIso = (ms) => new Date(ms ?? Date.now()).toISOString();
function newId(prefix) {
  const t = (/* @__PURE__ */ new Date()).toISOString().replace(/[-:.]/g, "").replace("T", "-").slice(0, 15);
  return `${prefix}-${t}-${Math.random().toString(36).slice(2, 8)}`;
}
var SANITIZE_MAX = 200;
function sanitize(value) {
  if (typeof value !== "string" && typeof value !== "number") return "";
  const s = String(value).replace(/[^A-Za-z0-9._:/@%+-]/g, "_");
  return s.length > SANITIZE_MAX ? s.slice(0, SANITIZE_MAX) : s;
}
var quoted = (key, value) => `${key}="${sanitize(value)}"`;
function typeOf(x) {
  if (x === null) return "null";
  if (Array.isArray(x)) return "array";
  if (typeof x === "number") return Number.isInteger(x) ? "integer" : "number";
  return typeof x;
}
function deref(node, root) {
  while (node && typeof node.$ref === "string") {
    const [file, frag] = node.$ref.split("#");
    if (file) root = file === "common.v1.schema.json" ? common_v1_schema_default : null;
    if (!root || !frag?.startsWith("/$defs/")) throw new Error(`unresolvable $ref ${node.$ref}`);
    node = root.$defs[frag.slice(7)];
  }
  return [node, root];
}
function validate(schema, value, root = schema, at = "") {
  const [s, r] = deref(schema, root);
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
function withDefaults(schema, value, root = schema) {
  const [s, r] = deref(schema, root);
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
    const st2 = await $.fs.stat(path);
    const value = JSON.parse(await $.fs.read(path));
    const probs = schemaName ? validate(SCHEMAS[schemaName], value) : [];
    return probs.length ? { value: null, problem: `${path}: ${probs[0]}`, mtimeMs: st2.mtimeMs } : { value, problem: null, mtimeMs: st2.mtimeMs };
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
  const st2 = await $.fs.stat(path);
  const offset = Math.max(0, st2.size - maxBytes);
  const text = offset > 0 ? await $.fs.read(path, { offset, length: st2.size - offset }) : await $.fs.read(path);
  const lines = text.split("\n").slice(offset > 0 ? 1 : 0, -1);
  return lines.flatMap((l) => {
    try {
      return l ? [JSON.parse(l)] : [];
    } catch {
      return [];
    }
  });
}
var DETACH_SCRIPT = 'log=$1; shift; setsid -f "$@" >>"$log" 2>&1 </dev/null';
var detachArgv = (log, argv) => ["sh", "-c", DETACH_SCRIPT, "sh", log, ...argv];

// plugins/agent-sys-o11y/mods/src/trigger/registry.ts
var DEAD_RETENTION_MS = 36e5;
var transcripts = /* @__PURE__ */ new Map();
function read($, path) {
  return $.fs.read(path).catch(() => null);
}
async function aliveOf($, pid, procStart) {
  const s = await read($, `/proc/${pid}/stat`);
  return s != null && (procStart == null || s.slice(s.lastIndexOf(")") + 2).split(" ")[19] === String(procStart));
}
async function tmuxServerOf($, pid) {
  const kv = (await read($, `/proc/${pid}/environ`))?.split("\0").find((x) => x.startsWith("TMUX="));
  return kv ? kv.slice(5).split(",")[0] : null;
}
async function transcriptOf($, configDir, sid) {
  if (transcripts.has(sid)) return transcripts.get(sid);
  for (const d of await $.fs.list(`${configDir}/projects`).catch(() => [])) {
    const p = `${configDir}/projects/${d.name}/${sid}.jsonl`;
    if (d.kind === "directory" && await $.fs.exists(p)) return transcripts.set(sid, p), p;
  }
  return null;
}
function tmuxMatches(recTmux, target) {
  if (!recTmux || typeof target !== "string" || !target) return false;
  const [sess, rest] = recTmux.split(":");
  return target === sess || target === recTmux || (rest ?? "").endsWith("." + target);
}
var statusOf = (s) => ["busy", "idle", "waiting", "shell"].includes(s) ? s : "unknown";
var under = (dir, root) => typeof dir === "string" && (dir === root || dir.startsWith(root.replace(/\/$/, "") + "/"));
async function normalise($, c, deadSince, now) {
  const dir = `${c.configDir}/sessions`;
  const names = (await $.fs.list(dir).catch(() => [])).map((x) => x.name).filter((n) => /^\d+\.json$/.test(n));
  const launches = (await readJsonl($, `${c.fleet}/${P.launches}`)).filter((l) => l && l.v === 1);
  const prev = (await readDoc($, `${c.fleet}/${P.sessions}`, "registry.sessions")).value?.sessions ?? [];
  const prevByPid = new Map(prev.map((s) => [s.pid, s]));
  const prevMains = new Set(prev.filter((s) => s.role === "main").map((s) => s.session_id));
  const pid = Number((await read($, "/proc/self/stat"))?.split(" ")[0]) || null;
  const isMe = (r) => r.sessionId === c.sessionId || r.pid === pid;
  const dead = {};
  const retained = (sid) => now - (dead[sid] = deadSince[sid] ?? now) <= DEAD_RETENTION_MS;
  const records = [];
  for (const n of names) {
    let r = null;
    try {
      r = JSON.parse(await read($, `${dir}/${n}`) ?? "");
    } catch {
    }
    if (r && typeof r.pid === "number" && typeof r.sessionId === "string") records.push(r);
    else if (prevByPid.has(Number(n.slice(0, -5)))) records.push({ __last: prevByPid.get(Number(n.slice(0, -5))) });
  }
  const me = records.find((r) => !r.__last && isMe(r));
  const myServer = pid ? await tmuxServerOf($, pid) : null;
  const sessions = [];
  const candidates = [];
  for (const r of records) {
    if (r.__last) {
      sessions.push({ ...r.__last, record_stale: true });
      continue;
    }
    const alive = await aliveOf($, r.pid, r.procStart);
    const join = launches.findLast((x) => x.session_id === r.sessionId || x.name && x.name === r.name || tmuxMatches(r.tmux ?? null, x.tmux_target));
    const mine = isMe(r);
    let role = mine ? "main" : "other";
    if (!mine && prevMains.has(r.sessionId)) {
      if (!alive && !retained(r.sessionId)) continue;
      role = "main";
    } else if (!mine && join?.decision === "child") role = "child";
    else if (!mine && !join && me && alive && r.tmux && under(r.cwd, c.launchDir) && (r.startedAt ?? 0) > (me.startedAt ?? 0)) {
      const server = await tmuxServerOf($, r.pid);
      if (!myServer || !server || server === myServer) candidates.push({ sid: r.sessionId, name: r.name ?? "", tmux: r.tmux, cwd: r.cwd });
    }
    if (role === "other") continue;
    const p = prevByPid.get(r.pid);
    const sid = mine ? c.sessionId : r.sessionId;
    const lineage = mine ? [...c.lineage] : p && p.session_id !== sid && p.proc_start === (r.procStart ?? null) ? [...p.lineage ?? [], p.session_id] : [...p?.lineage ?? []];
    sessions.push({
      session_id: sid,
      pid: r.pid,
      proc_start: r.procStart ?? null,
      alive,
      crashed: !alive,
      name: r.name ?? null,
      cwd: r.cwd ?? null,
      tmux: r.tmux ?? null,
      messaging_socket: typeof r.messagingSocketPath === "string" ? r.messagingSocketPath : null,
      transcript_path: await transcriptOf($, c.configDir, sid),
      status: statusOf(r.status),
      waiting_for: r.waitingFor ?? null,
      ...r.statusUpdatedAt ? { status_since: nowIso(r.statusUpdatedAt) } : {},
      role,
      parent_session_id: role === "child" ? join.parent_session_id ?? c.sessionId : null,
      lineage,
      ...r.startedAt ? { started_at: nowIso(r.startedAt) } : {},
      ...r.updatedAt ? { updated_at: nowIso(r.updatedAt) } : {},
      version: r.version ?? null,
      kind: r.kind ?? null,
      purpose: role === "child" ? join.purpose ?? null : null,
      active: mine,
      fleet_dir: c.fleet,
      config_dir: c.configDir,
      launch_dir: mine ? c.launchDir : null
    });
  }
  const listed = new Set(sessions.map((s) => s.session_id));
  for (const p of prev) {
    if (p.role === "main" && !listed.has(p.session_id) && p.pid !== pid && retained(p.session_id)) {
      sessions.push({ ...p, alive: false, active: false, crashed: p.alive === true || p.crashed === true });
    }
  }
  return { sessions, candidates, deadSince: dead };
}

// plugins/agent-sys-o11y/mods/src/trigger/schedule.ts
var MARK = { kick: "[o11y automated kick] ", packup: "[o11y automated packup] ", question: "[o11y automated question] " };
var LEDGER_TOOL = "mcp__plugin_agent-sys-o11y-door_o11y__ledger_append";
function fresh(now, cfg2) {
  return { next_kick_at: now + cfg2.kick.interval_s * 1e3, kicks: 0, kicks_since_packup: 0, last_packup_at: now };
}
function dueKick(s, cfg2, now) {
  if (now < s.next_kick_at) return null;
  let next_kick_at = s.next_kick_at;
  while (next_kick_at <= now) next_kick_at += cfg2.kick.interval_s * 1e3;
  const packup = s.kicks_since_packup + 1 >= cfg2.packup.every_n_kicks || now - s.last_packup_at >= cfg2.packup.interval_s * 1e3;
  return packup ? { kind: "packup", next: { next_kick_at, kicks: s.kicks + 1, kicks_since_packup: 0, last_packup_at: now } } : { kind: "kick", next: { ...s, next_kick_at, kicks: s.kicks + 1, kicks_since_packup: s.kicks_since_packup + 1 } };
}
function isPending(p, live, now, cfg2) {
  if (!p) return false;
  return live || now - p.submitted_at < 3 * cfg2.kick.interval_s * 1e3;
}
function render(template, vars) {
  return template.replace(/\{(\w+)\}/g, (m, k) => k in vars ? vars[k] : m);
}
function childrenText(sessions) {
  const kids = sessions.filter((s) => s.role === "child" && s.alive);
  if (!kids.length) return "none";
  return kids.map((s) => `[${quoted("session_id", s.session_id)} ${quoted("name", s.name)} ${quoted("tmux", s.tmux)}]`).join(" ");
}
function kickText(kind, cfg2, taskBook, sessions) {
  const vars = { task_book: quoted("path", sanitize(taskBook)), children: childrenText(sessions), ledger_tool: LEDGER_TOOL };
  return MARK[kind] + render(kind === "packup" ? cfg2.packup.template : cfg2.kick.template, vars);
}
function questionText(c) {
  return MARK.question + "A live session that is not registered in this fleet runs in your tmux server under the fleet directory. Either register it as your child with its purpose (session_register, decision child), or declare it not part of the fleet (session_register, decision not_in_fleet). Session data: " + [quoted("session_id", c.sid), quoted("name", c.name), quoted("tmux", c.tmux), quoted("cwd", c.cwd)].join(" ");
}
function pickCandidate(candidates, asked) {
  return candidates.find((c) => !asked.includes(c.sid)) ?? null;
}

// plugins/agent-sys-o11y/mods/src/trigger/counters.ts
function count(c, what, cfg2) {
  const counts = what === "tool" ? { ...c, tool_calls: c.tool_calls + 1 } : { ...c, turns: c.turns + 1 };
  const n = what === "tool" ? counts.tool_calls : counts.turns;
  const key = what === "tool" ? "every_n_tool_calls" : "every_n_turns";
  const kinds = [];
  if (n % cfg2.refresh[key] === 0) kinds.push("extract", "facts");
  if (n % cfg2.log[key] === 0) kinds.push("log");
  return { counts, kinds, reason: what === "tool" ? "tool_calls" : "turns" };
}

// plugins/agent-sys-o11y/mods/src/trigger/spawn.ts
var SPAWN_GAP_MS = 5e3;
var requests = null;
var chain = Promise.resolve();
var lastLaunch = {};
var deferred = {};
function request($, c, kind, reason) {
  const done = chain.then(() => bump($, c, kind, reason));
  chain = done.catch(() => {
  });
  return done;
}
async function bump($, c, kind, reason) {
  requests ??= (await readDoc($, `${c.fleet}/${P.requests}`, "requests")).value ?? { v: 1 };
  requests = { ...requests, [kind]: { seq: (requests[kind]?.seq ?? 0) + 1, ts: nowIso(), reason } };
  await writeDoc($, `${c.fleet}/${P.requests}`, requests);
  const wait = lastLaunch[kind] + SPAWN_GAP_MS - Date.now();
  if (!(wait > 0)) return start($, c, kind);
  if (!deferred[kind]) deferred[kind] = true, $.clock.after(wait, () => (deferred[kind] = false, start($, c, kind)));
  return null;
}
function start($, c, kind) {
  lastLaunch[kind] = Date.now();
  return launch($, c, [`${c.pluginRoot}/server/o11y-run`, "--kind", kind]);
}
async function launch($, c, argv) {
  try {
    const r = await $.process.run(detachArgv(`${c.fleet}/state/spawn.log`, argv), { env: c.env, timeoutMs: 1e4 });
    return r.exitCode === 0 ? null : `${argv[0]}: exit ${r.exitCode} ${String(r.stderr).slice(0, 200)}`;
  } catch (err) {
    return `${argv[0]}: ${err.message}`;
  }
}

// plugins/agent-sys-o11y/mods/src/trigger/otel_health.ts
var health = null;
async function sample($, c, url) {
  const healthy = await $.http.fetch(url).then((r) => r.ok === true, () => false);
  health ??= (await readDoc($, `${c.fleet}/${P.otelHealth}`, "otel_health")).value;
  const changed = !health || health.healthy !== healthy;
  const outages = (health?.outages ?? 0) + (health?.healthy !== false && !healthy ? 1 : 0);
  const last_change = changed ? nowIso() : health.last_change;
  const down_since = healthy ? void 0 : changed ? last_change : health.down_since;
  health = { v: 1, ts: nowIso(), healthy, outages, ...last_change ? { last_change } : {}, ...down_since ? { down_since } : {} };
  await writeDoc($, `${c.fleet}/${P.otelHealth}`, health);
}
async function sweep($, c) {
  const launcher = `${c.pluginRoot}/bin/o11y-launch`;
  if (await $.fs.exists(launcher)) await launch($, c, [launcher, "--sweep"]);
}

// plugins/agent-sys-o11y/mods/src/trigger/index.ts
var SKILL_RE = /^(infera-agent-sys|agent-sys-[\w-]+)$/;
var REGISTRY_TICK_MS = 3e4;
var sessionId = "";
var vetoed = false;
var active = false;
var env = {};
var reg = null;
var spawnCtx = null;
var cfgState = null;
var st = null;
var pendingLive = false;
var ticker = null;
var lastRegistry = "";
var chain2 = Promise.resolve();
var cfg = () => cfgState.cfg;
var abs = (rel) => `${reg.fleet}/${rel}`;
var serial = (fn) => chain2 = chain2.then(fn).catch(() => {
});
function skillName(raw) {
  const bare = typeof raw === "string" ? raw.replace(/^\//, "").split(/\s/)[0].split(":").pop() ?? "" : "";
  return SKILL_RE.test(bare) ? bare : null;
}
function saveState($) {
  const snap = JSON.parse(JSON.stringify(st));
  return serial(() => writeDoc($, abs(P.triggerState), snap));
}
function fire($, kind, outcome, x = {}) {
  const seq = ++st.seq;
  const rec = {
    v: 1,
    id: newId("trig"),
    ts: nowIso(),
    seq,
    kind,
    outcome,
    session_id: sessionId,
    ...x.detail ? { detail: x.detail.slice(0, 500) } : {},
    ...x.text ? { text: x.text } : {},
    ...x.subject ? { subject_session_id: x.subject } : {},
    counters: { tool_calls: st.counts.tool_calls, turns: st.counts.turns, kicks: st.schedule.kicks },
    writer: { session_id: sessionId, role: "main", source: "detected" }
  };
  return serial(() => writeDoc($, abs(`${P.triggers}/${String(seq).padStart(8, "0")}.json`), rec));
}
async function reloadConfig($) {
  const before = cfgState?.cfg.kick.interval_s;
  cfgState = await loadConfig($, abs(P.config("trigger")), "config.trigger", cfgState);
  if (cfgState.report) $.ui.log(`o11y: config/trigger.json not applied, the previous config stays: ${cfgState.report}`);
  if (st && before !== void 0 && cfg().kick.interval_s !== before) st.schedule.next_kick_at = Math.min(st.schedule.next_kick_at, Date.now() + cfg().kick.interval_s * 1e3);
}
async function submit($, kind, text, subject, detail) {
  st.pending = { kind, submitted_at: Date.now(), ...subject ? { subject } : {} };
  pendingLive = true;
  $.prompt.submit({ text }).then(
    (r) => r?.drop && (st.retry = { kind, text, subject }, fire($, kind, "refused", { detail: String(r.drop), subject })),
    (err) => fire($, kind, "failed", { detail: String(err?.message ?? err), subject })
  ).finally(() => (st.pending = null, saveState($)));
  await fire($, kind, "submitted", { text, subject, detail });
}
async function prompts($, now, sessions, candidates) {
  const due = dueKick(st.schedule, cfg(), now);
  if (due) {
    st.schedule = due.next;
    if (isPending(st.pending, pendingLive, now, cfg())) await fire($, due.kind, "skipped_pending", { detail: `${st.pending.kind} pending since ${nowIso(st.pending.submitted_at)}` });
    else await submit($, due.kind, kickText(due.kind, cfg(), cfg().task_book.startsWith("/") ? cfg().task_book : `${reg.launchDir}/${cfg().task_book}`, sessions));
  }
  const c = pickCandidate(candidates, st.asked);
  if (c && !isPending(st.pending, pendingLive, now, cfg())) {
    st.asked.push(c.sid);
    await submit($, "question", questionText(c), c.sid);
  }
}
function requestLater($, kinds, reason) {
  for (const k of kinds) $.clock.after(1, () => request($, spawnCtx, k, reason).then((p) => p && $.ui.log(`o11y: ${p}`, { to: "debug" })));
}
async function tick($) {
  if (!active || !st) return;
  try {
    await followClear($);
    const now = Date.now();
    await reloadConfig($);
    const { sessions, candidates, deadSince } = await normalise($, reg, st.dead_since, now);
    st.dead_since = deadSince;
    await writeRegistry($, sessions);
    await prompts($, now, sessions, candidates);
    if (now >= st.next_slow_at) {
      while (st.next_slow_at <= now) st.next_slow_at += cfg().log.slow_interval_s * 1e3;
      requestLater($, ["log"], "timer");
    }
    const url = await $.env.get("AGENT_SYS_O11Y_OTEL_HEALTH_URL");
    if (url) await sample($, spawnCtx, url).then(() => sweep($, spawnCtx));
    await saveState($);
  } catch (err) {
    $.ui.log(`o11y: trigger tick: ${err.message}`, { to: "debug" });
  }
}
async function writeRegistry($, sessions) {
  const body = JSON.stringify(sessions);
  if (body === lastRegistry) return;
  lastRegistry = body;
  await serial(() => writeDoc($, abs(P.sessions), { v: 1, generated_at: nowIso(), main_session_id: sessionId, sessions }));
}
function writePin($) {
  return writeDoc($, pinPath(env, sessionId), { v: 1, session_id: sessionId, fleet_dir: reg.fleet, launch_dir: reg.launchDir, pinned_at: nowIso(), lineage: reg.lineage });
}
async function activate($, by, pin) {
  if (active || vetoed) return;
  active = true;
  const launchDir = pin?.launch_dir ?? String(await $.session.root());
  const fleet = pin?.fleet_dir ?? await derivedFleetDir(env, launchDir);
  const configDir = configDirOf(env);
  reg = { fleet, launchDir, configDir, sessionId, lineage: pin?.lineage ?? [] };
  spawnCtx = { fleet, pluginRoot: $.plugin.root, env: { AGENT_SYS_O11Y_FLEET_DIR: fleet, CLAUDE_CONFIG_DIR: configDir, AGENT_SYS_HOME: prefixOf(env) } };
  await writePin($);
  await reloadConfig($);
  const s = (await readDoc($, abs(P.triggerState))).value;
  const now = Date.now();
  st = s?.v === 1 && s.session_id === sessionId ? { pending: null, retry: null, asked: [], dead_since: {}, ...s } : { v: 1, session_id: sessionId, seq: s?.seq ?? 0, schedule: fresh(now, cfg()), next_slow_at: now + cfg().log.slow_interval_s * 1e3, counts: { tool_calls: 0, turns: 0 }, pending: null, retry: null, asked: s?.asked ?? [], dead_since: s?.dead_since ?? {} };
  pendingLive = false;
  ticker?.cancel();
  ticker = $.clock.every(REGISTRY_TICK_MS, () => tick($));
  await tick($);
  $.ui.log(`o11y active (${by}); fleet ${fleet}`);
}
async function followClear($) {
  const sid = await $.session.id();
  if (!sid || sid === sessionId) return;
  const old = sessionId;
  sessionId = sid;
  if (!active || !reg) return;
  reg = { ...reg, sessionId: sid, lineage: [...reg.lineage, old] };
  st.session_id = sid;
  await writePin($);
}
function onSkill($, name, by) {
  const s = skillName(name);
  if (s && !active && !vetoed) $.clock.after(1, () => activate($, `${by}${s}`, null));
}
function register(on) {
  on("session.start", async ($, e, next) => {
    sessionId = await $.session.id();
    const flag = await $.env.get("AGENT_SYS_O11Y");
    vetoed = flag !== void 0 && flag !== "true";
    if (!vetoed) {
      env = await readEnv($);
      const pin = (await readDoc($, pinPath(env, sessionId), "session_pin")).value;
      if (flag === "true" || pin) await activate($, flag === "true" ? "env" : "re-armed", pin);
    }
    return next(e);
  });
  on("tool.call", async ($, e, next) => {
    const r = await next(e);
    if (e.agentId || vetoed) return r;
    if (!active) onSkill($, e.tool === "Skill" ? e.skill : null, "skill ");
    else {
      const c = count(st.counts, "tool", cfg());
      st.counts = c.counts;
      requestLater($, c.kinds, c.reason);
    }
    return r;
  });
  on("command.run", async ($, e, next) => {
    if (e.origin?.kind === "composer") onSkill($, e.command, "skill /");
    return next(e);
  });
  on("turn.start", async ($, e, next) => {
    if (active && st?.pending && !pendingLive && !e.agentId) st.pending = null;
    return next(e);
  });
  on("turn.complete", async ($, e, next) => {
    if (!active || !st || e.agentId) return next(e);
    const c = count(st.counts, "turn", cfg());
    st.counts = c.counts;
    requestLater($, c.kinds, c.reason);
    const r = st.retry;
    if (r && !e.isAborted) {
      st.retry = null;
      $.clock.after(1, () => submit($, r.kind, r.text, r.subject, "retry after refusal"));
    }
    return next(e);
  });
  on("session.end", async ($, e, next) => {
    if (active && st && e.reason !== "clear" && e.reason !== "resume") {
      ticker?.cancel();
      ticker = null;
      const ended = JSON.parse(lastRegistry || "[]").map((s) => s.session_id === sessionId ? { ...s, alive: false, active: false, crashed: false } : s);
      lastRegistry = "";
      await writeRegistry($, ended);
      if (await $.env.get("AGENT_SYS_O11Y_OTEL_HEALTH_URL")) void sweep($, spawnCtx);
    }
    if (active && st) void saveState($);
    return next(e);
  });
}
export {
  register
};
