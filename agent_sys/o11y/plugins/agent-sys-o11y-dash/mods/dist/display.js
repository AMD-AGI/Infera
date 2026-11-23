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
      $ref: "#/$defs/time"
    },
    enabled: {
      type: "boolean",
      default: true
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
        enabled: {
          type: "boolean",
          default: true
        },
        interval_s: {
          type: "number",
          default: 1800,
          minimum: 60
        },
        template: {
          type: "string",
          default: "Make sure you are actively pushing the work forward and not idle. Re-read the task book {task_book} and confirm the current work has not drifted from it. Record run starts/ends, plans and campaigns with the {ledger_tool} tool. Relay this check to the child sessions you manage: {children}."
        },
        while_busy: {
          type: "boolean",
          default: true,
          description: "queue the kick while the session is busy (D7)"
        }
      }
    },
    packup: {
      type: "object",
      additionalProperties: false,
      properties: {
        enabled: {
          type: "boolean",
          default: true
        },
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
    max_pending: {
      type: "integer",
      default: 1,
      minimum: 1,
      description: "at most this many submitted-but-not-started kicks; extras are skipped"
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
    },
    registry_interval_s: {
      type: "number",
      default: 30,
      minimum: 5
    },
    question: {
      type: "object",
      additionalProperties: false,
      properties: {
        enabled: {
          type: "boolean",
          default: true
        },
        template: {
          type: "string",
          default: "Session {name} (tmux {tmux}, cwd {cwd}) is not registered. Is it yours, and what is it for? Register it with session_register, or declare it not part of the fleet."
        }
      }
    },
    sanitize_max_chars: {
      type: "integer",
      default: 200,
      minimum: 20,
      description: "cap on each interpolated value in automated prompts (spec \xA74.1)"
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
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
      $ref: "#/$defs/time"
    },
    claude_bin: {
      type: "string",
      default: "claude",
      description: "claude executable for LLM passes"
    },
    log: {
      type: "object",
      additionalProperties: false,
      properties: {
        enabled: {
          type: "boolean",
          default: true
        },
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
        max_slice_bytes: {
          type: "integer",
          default: 2e5,
          minimum: 1e3,
          description: "upper bound of new transcript bytes sent per session per pass"
        },
        lookback_ledger_records: {
          type: "integer",
          default: 50,
          minimum: 0
        },
        timeout_s: {
          type: "number",
          default: 300,
          minimum: 10
        },
        lookback_bytes: {
          type: "integer",
          default: 8e5,
          minimum: 0,
          description: "first sight of a history: the model reads only its last N bytes"
        }
      }
    },
    display: {
      type: "object",
      additionalProperties: false,
      properties: {
        enabled: {
          type: "boolean",
          default: true
        },
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
          default: "prompts/display.md"
        },
        max_slice_bytes: {
          type: "integer",
          default: 5e4,
          minimum: 1e3
        },
        timeout_s: {
          type: "number",
          default: 120,
          minimum: 10
        },
        llm_fill_gaps: {
          type: "boolean",
          default: true,
          description: "use the LLM only when the ledger lacks current run / plan"
        },
        lookback_bytes: {
          type: "integer",
          default: 2e5,
          minimum: 0
        }
      }
    },
    detectors: {
      type: "object",
      additionalProperties: false,
      properties: {
        permission_wait_warn_s: {
          type: "number",
          default: 600,
          minimum: 1
        },
        idle_warn_s: {
          type: "number",
          default: 1200,
          minimum: 1
        },
        api_error_repeat: {
          type: "integer",
          default: 3,
          minimum: 1
        },
        token_step: {
          type: "integer",
          default: 1e6,
          minimum: 1
        },
        campaign_stale_s: {
          type: "number",
          default: 1800,
          minimum: 60
        },
        crash_grace_s: {
          type: "number",
          default: 60,
          minimum: 0
        }
      }
    },
    sources: {
      type: "object",
      additionalProperties: false,
      properties: {
        claude_config_dir: {
          type: [
            "string",
            "null"
          ],
          default: null,
          description: "null = per session: $CLAUDE_CONFIG_DIR, else ~/.claude"
        },
        machine_pool_dir: {
          type: [
            "string",
            "null"
          ],
          default: null,
          description: "machine-pool state dir (machine-pool skill); null = no machine-pool source"
        },
        campaign_globs: {
          type: "array",
          items: {
            type: "string"
          },
          default: []
        },
        otel_dir: {
          type: [
            "string",
            "null"
          ],
          default: null
        },
        e2e_flow_globs: {
          type: "array",
          items: {
            type: "string"
          },
          default: [],
          description: "where to look for e2e-flow kernel_table.json / integration_report.json"
        },
        task_book: {
          type: [
            "string",
            "null"
          ],
          default: null,
          description: "task book path; null = config.trigger task_book"
        }
      }
    },
    facts_stale_after_s: {
      type: "number",
      default: 900,
      minimum: 10
    },
    budget_s: {
      type: "number",
      default: 300,
      minimum: 10,
      description: "one analyzer invocation's runtime budget"
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
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
      $ref: "#/$defs/time"
    },
    enabled: {
      type: "boolean",
      default: true
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
    tab: {
      enum: [
        "general",
        "detail",
        "kernelforge",
        "log"
      ],
      default: "general"
    },
    band: {
      type: "boolean",
      default: true,
      description: "one-line band above the prompt"
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
    },
    log_min_level: {
      enum: [
        "INFO",
        "WARN",
        "ERROR"
      ],
      default: "INFO"
    },
    show_sessions: {
      type: "boolean",
      default: true
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
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
    "generated_at",
    "stale_after_s",
    "sources",
    "task",
    "best",
    "spend",
    "progress"
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
      },
      description: "files/sessions this document was built from"
    },
    problems: {
      type: "array",
      items: {
        type: "string"
      },
      description: "schema mismatches or missing sources, one line each"
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
          $ref: "#/$defs/time"
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
        wall_s: {
          type: "number",
          minimum: 0
        },
        started_at: {
          $ref: "#/$defs/time"
        },
        sessions: {
          type: "integer",
          minimum: 0
        },
        tokens_new: {
          type: "integer",
          minimum: 0
        },
        tokens_cache_read: {
          type: "integer",
          minimum: 0
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
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
};

// protocol/schemas/facts.detail.v1.schema.json
var facts_detail_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "facts.detail.v1.schema.json",
  title: "Display facts: detail tab",
  type: "object",
  required: [
    "v",
    "generated_at",
    "stale_after_s",
    "sources",
    "current_runs",
    "recent_runs",
    "next",
    "machines"
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
      },
      description: "files/sessions this document was built from"
    },
    problems: {
      type: "array",
      items: {
        type: "string"
      },
      description: "schema mismatches or missing sources, one line each"
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
            $ref: "#/$defs/time"
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
            $ref: "#/$defs/time"
          },
          ended_at: {
            $ref: "#/$defs/time"
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
            $ref: "#/$defs/time"
          },
          until: {
            $ref: "#/$defs/time"
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
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
};

// protocol/schemas/facts.kernelforge.v1.schema.json
var facts_kernelforge_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "facts.kernelforge.v1.schema.json",
  title: "Display facts: KernelForge tab",
  type: "object",
  required: [
    "v",
    "generated_at",
    "stale_after_s",
    "sources",
    "hot_ops",
    "campaigns",
    "ops"
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
      },
      description: "files/sessions this document was built from"
    },
    problems: {
      type: "array",
      items: {
        type: "string"
      },
      description: "schema mismatches or missing sources, one line each"
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
            $ref: "#/$defs/time"
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
            $ref: "#/$defs/time"
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
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
};

// protocol/schemas/facts.usage.v1.schema.json
var facts_usage_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "facts.usage.v1.schema.json",
  title: "Token usage per session and fleet (D1: incl. cache reads, deduped by message.id)",
  type: "object",
  required: [
    "v",
    "generated_at",
    "stale_after_s",
    "sources",
    "fleet",
    "sessions"
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
      },
      description: "files/sessions this document was built from"
    },
    problems: {
      type: "array",
      items: {
        type: "string"
      },
      description: "schema mismatches or missing sources, one line each"
    },
    fleet: {
      type: "object",
      required: [
        "tokens",
        "messages"
      ],
      properties: {
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
              description: "input + cache_creation + output (the milestone total)"
            }
          }
        },
        messages: {
          type: "integer",
          minimum: 0
        },
        steps_crossed: {
          type: "integer",
          minimum: 0
        }
      }
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
                description: "input + cache_creation + output (the milestone total)"
              }
            }
          },
          messages: {
            type: "integer",
            minimum: 0
          },
          subagent_files: {
            type: "integer",
            minimum: 0
          },
          last_at: {
            $ref: "#/$defs/time"
          },
          steps_crossed: {
            type: "integer",
            minimum: 0
          }
        }
      }
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
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
      $ref: "#/$defs/time"
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
            description: "registry file present, process gone"
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
            $ref: "#/$defs/time"
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
            $ref: "#/$defs/time"
          },
          updated_at: {
            $ref: "#/$defs/time"
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
          }
        }
      }
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
};

// protocol/schemas/registry.launch.v1.schema.json
var registry_launch_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "registry.launch.v1.schema.json",
  title: "Session registration (registry/launches.jsonl), written by session_register",
  type: "object",
  required: [
    "v",
    "id",
    "ts",
    "role",
    "name",
    "writer"
  ],
  properties: {
    v: {
      const: 1
    },
    id: {
      $ref: "#/$defs/id"
    },
    ts: {
      $ref: "#/$defs/time"
    },
    role: {
      enum: [
        "main",
        "child",
        "declined"
      ],
      description: "declined = a session the main said is not part of the fleet (\xA73.5)"
    },
    name: {
      type: "string",
      minLength: 1
    },
    session_id: {
      type: "string",
      description: "set when a session registers itself"
    },
    parent_session_id: {
      type: "string"
    },
    tmux_target: {
      type: "string"
    },
    purpose: {
      type: "string"
    },
    cwd: {
      type: "string"
    },
    writer: {
      $ref: "#/$defs/writer"
    },
    fleet_dir: {
      type: "string"
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
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
      $ref: "#/$defs/id"
    },
    ts: {
      $ref: "#/$defs/time"
    },
    seq: {
      type: "integer",
      minimum: 0
    },
    kind: {
      enum: [
        "kick",
        "packup",
        "question",
        "refresh",
        "log"
      ]
    },
    outcome: {
      enum: [
        "submitted",
        "queued",
        "skipped_pending",
        "refused",
        "dropped",
        "enqueued",
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
      $ref: "#/$defs/writer"
    },
    text: {
      type: "string",
      description: "exact text handed to Claude Code"
    },
    subject_session_id: {
      type: "string",
      description: "question: the unregistered session"
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
};

// protocol/schemas/queue.v1.schema.json
var queue_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "queue.v1.schema.json",
  title: "Work item from trigger to analysis (queue/<ts>-<kind>.json)",
  type: "object",
  required: [
    "v",
    "id",
    "ts",
    "kind",
    "reason",
    "writer"
  ],
  properties: {
    v: {
      const: 1
    },
    id: {
      $ref: "#/$defs/id"
    },
    ts: {
      $ref: "#/$defs/time"
    },
    kind: {
      enum: [
        "display",
        "log"
      ]
    },
    reason: {
      enum: [
        "tool_calls",
        "turns",
        "timer",
        "manual",
        "session_event"
      ]
    },
    session_id: {
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
        }
      }
    },
    writer: {
      $ref: "#/$defs/writer"
    }
  },
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    }
  }
};

// protocol/schemas/event.v1.schema.json
var event_v1_schema_default = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "event.v1.schema.json",
  title: "o11y log record (OTel Log Data Model names, flattened)",
  type: "object",
  required: [
    "id",
    "v",
    "ts",
    "severity_text",
    "severity_number",
    "event_name",
    "body",
    "attributes",
    "writer"
  ],
  properties: {
    id: {
      $ref: "#/$defs/id"
    },
    v: {
      const: 1
    },
    ts: {
      $ref: "#/$defs/time"
    },
    observed_ts: {
      $ref: "#/$defs/time",
      description: "when the underlying fact happened, if different"
    },
    severity_text: {
      enum: [
        "INFO",
        "WARN",
        "ERROR"
      ]
    },
    severity_number: {
      enum: [
        9,
        13,
        17
      ]
    },
    event_name: {
      type: "string",
      pattern: "^o11y\\.[a-z0-9_]+(\\.[a-z0-9_]+)+$",
      description: "o11y.<area>.<what>, e.g. o11y.session.crashed, o11y.tokens.crossed"
    },
    body: {
      type: "string",
      minLength: 1,
      maxLength: 2e3,
      description: "one human line"
    },
    attributes: {
      type: "object",
      required: [
        "o11y.source",
        "o11y.dedup_key"
      ],
      properties: {
        "session.id": {
          type: "string"
        },
        "session.name": {
          type: "string"
        },
        "session.role": {
          enum: [
            "main",
            "child",
            "analyzer",
            "other"
          ]
        },
        "o11y.source": {
          enum: [
            "detector",
            "llm",
            "agent",
            "otel"
          ]
        },
        "o11y.dedup_key": {
          type: "string",
          minLength: 1,
          description: "same key = same underlying fact across passes (spec \xA78.5)"
        },
        evidence: {
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
              offset: {
                type: "integer",
                minimum: 0
              },
              line: {
                type: "integer",
                minimum: 1
              },
              uuid: {
                type: "string"
              }
            }
          }
        }
      },
      additionalProperties: true
    },
    writer: {
      $ref: "#/$defs/writer"
    }
  },
  allOf: [
    {
      if: {
        properties: {
          severity_text: {
            const: "INFO"
          }
        }
      },
      then: {
        properties: {
          severity_number: {
            const: 9
          }
        }
      }
    },
    {
      if: {
        properties: {
          severity_text: {
            const: "WARN"
          }
        }
      },
      then: {
        properties: {
          severity_number: {
            const: 13
          }
        }
      }
    },
    {
      if: {
        properties: {
          severity_text: {
            const: "ERROR"
          }
        }
      },
      then: {
        properties: {
          severity_number: {
            const: 17
          }
        }
      }
    }
  ],
  $defs: {
    time: {
      type: "string",
      format: "date-time",
      pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})$",
      description: "RFC 3339 timestamp"
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
    id: {
      type: "string",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
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
  "facts.usage": facts_usage_v1_schema_default,
  "registry.sessions": registry_sessions_v1_schema_default,
  "registry.launch": registry_launch_v1_schema_default,
  trigger_fire: trigger_fire_v1_schema_default,
  queue: queue_v1_schema_default,
  event: event_v1_schema_default
};
var P = {
  config: (name) => `config/${name}.json`,
  sessions: "registry/sessions.json",
  launches: "registry/launches.jsonl",
  events: "logs/events.jsonl",
  triggers: "triggers",
  queue: "queue",
  facts: (tab2) => `facts/${tab2}.json`,
  triggerState: "state/trigger.json",
  active: (sid) => `state/active/${sid}.json`
};
async function slug(dir) {
  const name = dir.replace(/[^A-Za-z0-9]/gu, "-");
  if (name.length <= 200) return name;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(dir)));
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${name.slice(0, 200)}-${hex.slice(0, 12)}`;
}
function trimSlash(p) {
  return p.length > 1 ? p.replace(/\/+$/, "") : p;
}
function prefixOf(env) {
  const h = env.AGENT_SYS_HOME;
  if (h && h.startsWith("/")) return trimSlash(h);
  return `${trimSlash(env.HOME || "/tmp")}/.infera_agent_sys`;
}
async function fleetDirOf(env, launchDir) {
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
function nowIso(ms) {
  return new Date(ms ?? Date.now()).toISOString();
}
function typeOf(x) {
  if (x === null) return "null";
  if (Array.isArray(x)) return "array";
  if (typeof x === "number") return Number.isInteger(x) ? "integer" : "number";
  return typeof x;
}
function resolve(root2, node) {
  if (node && typeof node.$ref === "string" && node.$ref.startsWith("#/$defs/")) {
    return resolve(root2, root2.$defs[node.$ref.slice(8)]);
  }
  return node;
}
function validate(schema, value, root2 = schema, at = "") {
  const s = resolve(root2, schema);
  if (!s || typeof s !== "object") return [];
  const where = at || "/";
  const t = typeOf(value);
  if (s.type) {
    const types = Array.isArray(s.type) ? s.type : [s.type];
    if (!types.some((ty) => ty === t || ty === "number" && t === "integer")) {
      return [`${where}: expected ${types.join("|")}, got ${t}`];
    }
  }
  const out = [];
  if (s.const !== void 0 && value !== s.const) out.push(`${where}: must be ${JSON.stringify(s.const)}`);
  if (s.enum && !s.enum.includes(value)) out.push(`${where}: must be one of ${s.enum.join(", ")}`);
  if (typeof value === "number") {
    if (typeof s.minimum === "number" && value < s.minimum) out.push(`${where}: must be >= ${s.minimum}`);
    if (typeof s.maximum === "number" && value > s.maximum) out.push(`${where}: must be <= ${s.maximum}`);
  }
  if (typeof value === "string") {
    if (typeof s.minLength === "number" && value.length < s.minLength) out.push(`${where}: too short`);
    if (typeof s.pattern === "string" && !new RegExp(s.pattern).test(value)) out.push(`${where}: does not match ${s.pattern}`);
  }
  if (t === "object") {
    const obj = value;
    for (const k of s.required ?? []) if (!(k in obj)) out.push(`${at}/${k}: required`);
    for (const [k, v] of Object.entries(obj)) {
      const ps = s.properties?.[k];
      if (ps) out.push(...validate(ps, v, root2, `${at}/${k}`));
      else if (s.additionalProperties === false) out.push(`${at}/${k}: unknown key`);
    }
  }
  if (t === "array") {
    const arr = value;
    if (typeof s.minItems === "number" && arr.length < s.minItems) out.push(`${where}: needs ${s.minItems}+ items`);
    if (s.items) arr.forEach((v, i) => out.push(...validate(s.items, v, root2, `${at}/${i}`)));
  }
  return out;
}
function withDefaults(schema, value, root2 = schema) {
  const s = resolve(root2, schema);
  if (!s || s.type !== "object" || !s.properties) return value;
  const out = { ...value ?? {} };
  for (const [k, ps0] of Object.entries(s.properties)) {
    const ps = resolve(root2, ps0);
    if (out[k] === void 0 && ps.default !== void 0) out[k] = JSON.parse(JSON.stringify(ps.default));
    if (ps.type === "object" && ps.properties) out[k] = withDefaults(ps, out[k], root2);
  }
  return out;
}
async function readDoc($, path, schemaName) {
  try {
    if (!await $.fs.exists(path)) return { value: null, problem: null, mtimeMs: 0 };
    const st = await $.fs.stat(path);
    const value = JSON.parse(await $.fs.read(path));
    const probs = schemaName ? validate(SCHEMAS[schemaName], value) : [];
    return { value, problem: probs.length ? `${path}: ${probs[0]}` : null, mtimeMs: st.mtimeMs };
  } catch (err) {
    return { value: null, problem: `${path}: ${err.message}`, mtimeMs: 0 };
  }
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
  const text = offset > 0 ? await $.fs.read(path, { offset, length: st.size - offset }) : await $.fs.read(path);
  const lines = text.split("\n");
  if (offset > 0) lines.shift();
  lines.pop();
  const out = [];
  for (const l of lines) {
    if (!l) continue;
    try {
      out.push(JSON.parse(l));
    } catch {
    }
  }
  return out;
}

// plugins/agent-sys-o11y-dash/mods/src/display/index.ts
var PANE = "o11y";
var MENU = "o11y-menu";
var FORM = "o11y-cfg";
var TABS = ["general", "detail", "kernelforge", "log"];
var FACTS = ["general", "detail", "kernelforge", "usage"];
var SIZE = { narrow: { columns: 46, rows: 16 }, wide: { columns: 96, rows: 40 } };
var COMMANDS = [
  { name: "o11y", description: "o11y: menu (g/d/k/l tabs, m mode, b band, t/a/v config)" },
  { name: "o11y-general", description: "o11y: general tab" },
  { name: "o11y-detail", description: "o11y: detail tab (runs, plan, machines)" },
  { name: "o11y-kf", description: "o11y: KernelForge tab" },
  { name: "o11y-log", description: "o11y: log tab" },
  { name: "o11y-mode", description: "o11y: switch NARROW / WIDE", argumentHint: "[narrow|wide]" },
  { name: "o11y-band", description: "o11y: band above the prompt on/off", argumentHint: "[on|off]" },
  { name: "o11y-cfg-trigger", description: "o11y: edit trigger config (kick, packup, frequencies)" },
  { name: "o11y-cfg-analysis", description: "o11y: edit analysis config (model, effort, lookback)" },
  { name: "o11y-cfg-display", description: "o11y: edit display config" }
];
var root = null;
var vetoReason = "";
var sessionId = "";
var active = false;
var registryKey = "";
var cfg = withDefaults(SCHEMAS["config.display"], { v: 1 });
var cfgMtime = -1;
var tab = "general";
var mode = "narrow";
var bandOn = true;
var isOpen = false;
var docs = {};
var sessions = [];
var sessionsMtime = -1;
var events = [];
var eventsSize = -1;
var lastFire = null;
var lastFireName = "";
var polls = 0;
var poller = null;
var pollMs = 0;
var formTarget = "trigger";
var formDraft = {};
var formNote = "";
function abs(rel) {
  return `${root}/${rel}`;
}
async function checkActive($) {
  sessionId = await $.session.id() || sessionId;
  const sp = abs(P.sessions);
  const m = await $.fs.exists(sp) ? (await $.fs.stat(sp)).mtimeMs : 0;
  const key = `${m}:${sessionId}`;
  if (key === registryKey) return;
  registryKey = key;
  const reg = (await readDoc($, sp, "registry.sessions")).value;
  sessions = reg?.sessions ?? [];
  sessionsMtime = m;
  const mine = sessions.some((x) => x.session_id === sessionId && x.role === "main" && x.alive && x.active === true);
  if (mine && !active) {
    active = true;
    await poll($);
    mode = cfg.mode;
    tab = cfg.tab;
    bandOn = cfg.band;
    await openPane($, false);
    $.ui.invalidate("ui.render");
  } else if (!mine && active) {
    active = false;
    await $.ui.close({ id: PANE });
    isOpen = false;
    $.ui.invalidate("ui.render");
  }
}
async function poll($) {
  if (!root) return;
  let changed = false;
  try {
    await checkActive($);
    if (!active) return;
    const c = await readDoc($, abs(P.config("display")), "config.display");
    if (c.mtimeMs !== cfgMtime) {
      cfgMtime = c.mtimeMs;
      cfg = withDefaults(SCHEMAS["config.display"], c.problem ? { v: 1 } : c.value ?? { v: 1 });
      if (c.problem) docs["config"] = { value: null, problem: c.problem, mtimeMs: c.mtimeMs };
      else delete docs["config"];
      changed = true;
    }
    for (const f of FACTS) {
      const path = abs(P.facts(f));
      const exists = await $.fs.exists(path);
      const m = exists ? (await $.fs.stat(path)).mtimeMs : 0;
      if (docs[f]?.mtimeMs === m && docs[f] !== void 0) continue;
      const r = await readDoc($, path, `facts.${f}`);
      docs[f] = r.problem ? { ...r, value: null, problem: "invalid " + r.problem.replace(root + "/", "") } : r;
      changed = true;
    }
    const ep = abs(P.events);
    const es = await $.fs.exists(ep) ? (await $.fs.stat(ep)).size : 0;
    if (es !== eventsSize) {
      eventsSize = es;
      events = (await readJsonl($, ep, 256 * 1024)).filter((x) => x && x.v === 1 && x.severity_text);
      changed = true;
    }
    if (polls++ % 5 === 0 && await $.fs.exists(abs(P.triggers))) {
      const names = (await $.fs.list(abs(P.triggers))).map((x) => x.name).filter((n) => /^\d+\.json$/.test(n));
      const newest = names.sort().pop() ?? "";
      if (newest && newest !== lastFireName) {
        lastFireName = newest;
        lastFire = (await readDoc($, abs(`${P.triggers}/${newest}`), "trigger_fire")).value;
        changed = true;
      }
    }
  } catch (err) {
    $.ui.log(`o11y display poll: ${err.message}`, { to: "debug" });
  }
  if (cfg.poll_ms !== pollMs) startPoller($);
  if (changed) $.ui.invalidate("ui.render");
}
function startPoller($) {
  if (poller) poller.cancel();
  pollMs = cfg.poll_ms;
  poller = $.clock.every(pollMs, () => poll($));
}
function num(x, digits = 0) {
  if (typeof x !== "number" || !isFinite(x)) return "-";
  return x.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
function pct(x) {
  return typeof x === "number" && isFinite(x) ? `${x >= 0 ? "+" : ""}${x.toFixed(1)}%` : "-";
}
function tok(n) {
  if (typeof n !== "number") return "-";
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n);
}
function dur(s) {
  if (typeof s !== "number" || !isFinite(s)) return "-";
  const h = Math.floor(s / 3600);
  const m = Math.floor(s % 3600 / 60);
  return h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m`;
}
function ago(iso) {
  const t = Date.parse(iso);
  return isNaN(t) ? "-" : dur(Math.max(0, (Date.now() - t) / 1e3));
}
function staleOf(name) {
  const d = docs[name]?.value;
  if (!d) return docs[name]?.problem ? "invalid" : "no data";
  const age = (Date.now() - Date.parse(d.generated_at)) / 1e3;
  return age > d.stale_after_s ? `stale ${dur(age)}` : null;
}
function generalLines(wide) {
  const g = docs.general?.value;
  const u = docs.usage?.value;
  if (!g) return [[docs.general?.problem ?? "no facts/general.json yet", "yellow"]];
  const out = [];
  const t = g.task;
  if (t) {
    if (wide) out.push([`${t.title}${t.phase ? ` [${t.phase}]` : ""}`]);
    out.push([`${t.metric.name} (${t.metric.unit}, ${t.metric.direction} is better)`]);
    out.push([`base ${num(t.baseline.value)}${t.baseline.label && wide ? ` (${t.baseline.label})` : ""} \xB7 target ${t.target ? num(t.target.value) : "-"}`]);
  } else out.push(["task: not recorded yet (ledger kind=task)", "yellow"]);
  const b = g.best;
  if (b) {
    out.push([`best ${num(b.value)}  ${pct(b.vs_baseline_pct)} vs base \xB7 ${b.of_target_pct == null ? "-" : b.of_target_pct.toFixed(1) + "%"} of target`, "green"]);
    if (wide) out.push([`  runs ${b.run_ids.join(", ")}${b.stack ? ` \xB7 ${b.stack}` : ""}${b.at ? ` \xB7 ${ago(b.at)} ago` : ""}`]);
  }
  const s = g.spend;
  out.push([`tokens ${tok(s.tokens_new)} new + ${tok(s.tokens_cache_read)} cache reads \xB7 wall ${dur(s.wall_s)}${s.sessions ? ` \xB7 ${s.sessions} sessions` : ""}`]);
  const p = g.progress;
  out.push([`progress ${pct(p.gain_pct)} \xB7 ${pct(p.per_hour_pct)}/h \xB7 ${pct(p.per_mtok_pct)}/Mtok`]);
  if (wide) {
    out.push([`  abs ${num(p.gain_abs, 1)} \xB7 ${num(p.per_hour_abs, 1)}/h \xB7 ${num(p.per_mtok_abs, 1)}/Mtok`]);
    if (u) {
      const k = u.fleet.tokens;
      out.push([`  in ${tok(k.input)} \xB7 cache w ${tok(k.cache_creation)} r ${tok(k.cache_read)} \xB7 out ${tok(k.output)}`]);
    }
    if (cfg.show_sessions) {
      out.push(["sessions:"]);
      for (const x of sessions) {
        const ut = u?.sessions?.find((y) => y.session_id === x.session_id)?.tokens?.new;
        out.push([`  ${x.role === "main" ? "*" : " "} ${x.name ?? x.session_id.slice(0, 8)} ${x.alive ? x.status : "DEAD"}${x.waiting_for ? ` (${x.waiting_for})` : ""} \xB7 ${tok(ut)}`, x.alive ? void 0 : "red"]);
      }
    }
    for (const pr of g.problems ?? []) out.push([`! ${pr}`, "yellow"]);
  }
  return out;
}
function detailLines(wide) {
  const d = docs.detail?.value;
  if (!d) return [[docs.detail?.problem ?? "no facts/detail.json yet", "yellow"]];
  const out = [];
  for (const r of d.current_runs) {
    const m = (r.machines ?? []).map((x) => typeof x === "string" ? x : x.node).join(",");
    out.push([`\u25B6 ${r.run_id} ${r.theme}${r.point ? ` [${r.point}]` : ""}${m ? ` @${m}` : ""}`, "cyan"]);
    if (wide && r.hypothesis) out.push([`    hyp: ${r.hypothesis}`]);
  }
  if (!d.current_runs.length) out.push(["no run in progress", "dim"]);
  out.push([`last ${Math.min(cfg.recent_runs_n, d.recent_runs.length)} runs:`]);
  for (const r of d.recent_runs.slice(0, cfg.recent_runs_n)) {
    const c = r.new_best ? "green" : r.status === "failed" || r.status === "rejected" ? "red" : void 0;
    out.push([`  ${r.status.padEnd(8)} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, c]);
    if (wide && r.conclusion) out.push([`      ${r.conclusion}`]);
  }
  out.push([`next: ${d.next.length ? d.next[0].theme : "-"}`]);
  if (wide) for (const n of d.next.slice(1)) out.push([`      ${n.theme}${n.blocked_on ? ` (blocked: ${n.blocked_on})` : ""}`]);
  out.push([`machines (${d.machines.length}):`]);
  for (const m of wide ? d.machines : d.machines.slice(0, 4)) {
    out.push([`  ${m.node}${m.cluster && wide ? `/${m.cluster}` : ""} \u2192 ${m.runs ?? "idle"}${wide && m.gpu_util != null ? ` \xB7 gpu ${m.gpu_util}%` : ""}`]);
  }
  if (wide) for (const pr of d.problems ?? []) out.push([`! ${pr}`, "yellow"]);
  return out;
}
function kfLines(wide) {
  const k = docs.kernelforge?.value;
  if (!k) return [[docs.kernelforge?.problem ?? "no facts/kernelforge.json yet", "yellow"]];
  const out = [];
  for (const h of k.hot_ops.slice(0, wide ? 3 : 1)) {
    out.push([`hot ops (${h.role}${h.scope && wide ? `, ${h.scope}` : ""}):`]);
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
var LEVEL = { INFO: 9, WARN: 13, ERROR: 17 };
var COLOR = { INFO: void 0, WARN: "yellow", ERROR: "red" };
function logLines(wide) {
  const min = LEVEL[cfg.log_min_level] ?? 9;
  const ev = events.filter((x) => (x.severity_number ?? LEVEL[x.severity_text] ?? 9) >= min).slice(-cfg.log_lines);
  if (!ev.length) return [["no events yet", "dim"]];
  return ev.map((x) => {
    const t = (x.ts ?? "").slice(11, 19);
    const who = wide && x.attributes?.["session.name"] ? ` ${x.attributes["session.name"]}` : "";
    return [`${t} ${x.severity_text.padEnd(5)}${wide ? ` ${x.event_name}` : ""}${who} ${x.body}`, COLOR[x.severity_text]];
  });
}
function linesOf(t, wide) {
  if (t === "general") return generalLines(wide);
  if (t === "detail") return detailLines(wide);
  if (t === "kernelforge") return kfLines(wide);
  return logLines(wide);
}
function factFor(t) {
  return t === "log" ? null : t;
}
async function openPane($, focus) {
  const size = SIZE[mode];
  const r = await $.ui.open(focus ? { id: PANE, title: "o11y", focus: true, ...size } : { id: PANE, title: "o11y", ...size });
  isOpen = true;
  return r;
}
async function dRenderPane($, e, next) {
  if (e.requestId === MENU) return renderMenu($, e);
  if (e.requestId === FORM) return renderForm($, e);
  if (e.requestId !== PANE || !active) return next(e);
  const { Box, Text, Button } = $.ui.resolve(e);
  const wide = mode === "wide";
  const tabRow = Box({
    flexDirection: "row",
    columnGap: 2,
    children: [
      ...TABS.map(
        (t, i) => Button({
          key: `tab-${t}`,
          label: t === "kernelforge" && !wide ? "kf" : t,
          hotkey: String(i + 1),
          plain: true,
          dimColor: t !== tab,
          onPress: () => selectTab($, t)
        })
      ),
      Button({ key: "tab-mode", label: wide ? "narrow" : "wide", hotkey: "m", plain: true, dimColor: true, onPress: () => setMode($, wide ? "narrow" : "wide") })
    ]
  });
  const f = factFor(tab);
  const stale = f ? staleOf(f) : null;
  const meta = f && docs[f]?.value ? `${stale ? "\u26A0 " + stale : "updated " + ago(docs[f].value.generated_at) + " ago"}` : "";
  const body = linesOf(tab, wide).map(
    ([s, color], i) => Text({ key: `l${i}`, wrap: "truncate-end", ...color === "dim" ? { dimColor: true } : color ? { color } : {}, children: [s] })
  );
  return Box({
    flexDirection: "column",
    children: [tabRow, ...meta ? [Text({ wrap: "truncate-end", ...stale ? { color: "yellow" } : { dimColor: true }, children: [meta] })] : [], ...body]
  });
}
async function dRenderBand($, e, next) {
  if (!active || !bandOn || !cfg.band) return next(e);
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
function selectTab($, t) {
  tab = t;
  $.ui.invalidate("ui.render");
}
async function setMode($, m) {
  mode = m;
  await openPane($, false);
  $.ui.invalidate("ui.render");
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
  const btn = ([k, label]) => Button({ key: `menu-${k}`, label, hotkey: k, plain: true, onPress: () => menuPick($, k) });
  return Box({
    flexDirection: "column",
    children: [
      Text({ dimColor: true, children: ["o11y: press a key, Esc to cancel"] }),
      Box({ flexDirection: "row", columnGap: 2, children: items.slice(0, 5).map(btn) }),
      Box({ flexDirection: "row", columnGap: 2, children: items.slice(5).map(btn) })
    ]
  });
}
async function menuPick($, k) {
  await $.ui.close({ id: MENU });
  const tabs = { g: "general", d: "detail", k: "kernelforge", l: "log" };
  if (tabs[k]) return showTab($, tabs[k]);
  if (k === "m") return setMode($, mode === "wide" ? "narrow" : "wide");
  if (k === "b") {
    bandOn = !bandOn;
    $.ui.invalidate("ui.render");
    return;
  }
  if (k === "t") return openForm($, "trigger");
  if (k === "a") return openForm($, "analysis");
  if (k === "v") return openForm($, "display");
  if (k === "x") {
    await $.ui.close({ id: PANE });
    isOpen = false;
  }
}
async function showTab($, t) {
  tab = t;
  await openPane($, false);
  $.ui.invalidate("ui.render");
}
function fieldsOf(schema, at = []) {
  const out = [];
  for (const [k, ps] of Object.entries(schema.properties ?? {})) {
    if (k === "v" || k === "updated_at") continue;
    if (ps.type === "object" && ps.properties) out.push(...fieldsOf(ps, [...at, k]));
    else out.push({ path: [...at, k], schema: ps });
  }
  return out;
}
function getAt(obj, path) {
  return path.reduce((o, k) => o == null ? void 0 : o[k], obj);
}
function setAt(obj, path, value) {
  let o = obj;
  for (const k of path.slice(0, -1)) o = o[k] ??= {};
  o[path[path.length - 1]] = value;
}
async function openForm($, target) {
  if (!active) return;
  formTarget = target;
  const r = await readDoc($, abs(P.config(target)));
  formDraft = r.value && typeof r.value === "object" ? r.value : { v: 1 };
  formNote = r.problem ? `current file invalid: ${r.problem}` : `config/${target}.json`;
  await $.ui.open({ id: FORM, title: `o11y config: ${target}`, focus: true, closeOnEscape: true, rows: 30, columns: 90 });
}
function parseField(f, raw) {
  const t = Array.isArray(f.schema.type) ? f.schema.type[0] : f.schema.type;
  if (t === "number" || t === "integer") {
    const n = Number(raw);
    if (raw.trim() === "" || !isFinite(n) || t === "integer" && !Number.isInteger(n)) return { ok: false, problem: `${f.path.join(".")}: not a ${t}` };
    return { ok: true, value: n };
  }
  if (t === "array") {
    try {
      return { ok: true, value: JSON.parse(raw) };
    } catch {
      return { ok: false, problem: `${f.path.join(".")}: not a JSON array` };
    }
  }
  return { ok: true, value: raw };
}
async function saveField($, f, value) {
  const draft = JSON.parse(JSON.stringify(formDraft));
  setAt(draft, f.path, value);
  draft.v = 1;
  draft.updated_at = nowIso();
  const probs = validate(SCHEMAS[`config.${formTarget}`], draft);
  if (probs.length) {
    formNote = `not saved: ${probs[0]}`;
  } else {
    await writeDoc($, abs(P.config(formTarget)), draft);
    formDraft = draft;
    formNote = `saved ${f.path.join(".")} = ${JSON.stringify(value).slice(0, 60)}`;
  }
  $.ui.invalidate("ui.render");
}
async function renderForm($, e) {
  const { Box, Text, Input, Select } = $.ui.resolve(e);
  const schema = SCHEMAS[`config.${formTarget}`];
  const full = withDefaults(schema, formDraft);
  const rows = fieldsOf(schema).map((f, i) => {
    const key = `cfg-${formTarget}-${f.path.join(".")}`;
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
        onSelect: (v) => saveField($, f, isBool ? v === "true" : v)
      });
    }
    return Input({
      key,
      label,
      value: f.schema.type === "array" ? JSON.stringify(cur ?? []) : String(cur ?? ""),
      submitLabel: "save",
      ...i === 0 ? { autoFocus: true } : {},
      onSubmit: (raw) => {
        const p = parseField(f, raw);
        if (!p.ok) {
          formNote = `not saved: ${p.problem}`;
          $.ui.invalidate("ui.render");
          return;
        }
        return saveField($, f, p.value);
      }
    });
  });
  return Box({
    flexDirection: "column",
    children: [Text({ dimColor: true, wrap: "truncate-end", children: [`${formNote} \xB7 Tab/Enter to edit, Esc to close`] }), ...rows]
  });
}
async function command($, name, args) {
  if (!active) {
    const why = vetoReason || `this session is not the active main of fleet ${root} (it becomes one with AGENT_SYS_O11Y=true or on first agent_sys skill use; the display follows within one trigger tick)`;
    return { text: `o11y display inactive: ${why}`, isTextDisplayOnly: true };
  }
  const a = (args ?? "").trim();
  switch (name) {
    case "o11y":
      await $.ui.open({ id: MENU, title: "o11y menu", focus: true, closeOnEscape: true, holdToasts: true, rows: 4 });
      break;
    case "o11y-general":
      await showTab($, "general");
      break;
    case "o11y-detail":
      await showTab($, "detail");
      break;
    case "o11y-kf":
      await showTab($, "kernelforge");
      break;
    case "o11y-log":
      await showTab($, "log");
      break;
    case "o11y-mode":
      await setMode($, a === "narrow" || a === "wide" ? a : mode === "wide" ? "narrow" : "wide");
      break;
    case "o11y-band":
      bandOn = a === "on" ? true : a === "off" ? false : !bandOn;
      $.ui.invalidate("ui.render");
      break;
    case "o11y-cfg-trigger":
      await openForm($, "trigger");
      break;
    case "o11y-cfg-analysis":
      await openForm($, "analysis");
      break;
    case "o11y-cfg-display":
      await openForm($, "display");
      break;
  }
  return {};
}
function register(on) {
  on("session.start", async ($, e, next) => {
    sessionId = await $.session.id();
    for (const c of COMMANDS) {
      try {
        await $.command.register({ ...c, immediate: true });
      } catch (err) {
        $.ui.log(`o11y: /${c.name} not registered: ${err.message}`);
      }
    }
    const env = await $.env.get("AGENT_SYS_O11Y");
    if (env !== void 0 && env !== "true") {
      vetoReason = `AGENT_SYS_O11Y=${env}; the display runs only in a fleet's main session`;
      return next(e);
    }
    root = await fleetDirOf(await readEnv($), String(await $.session.root()));
    startPoller($);
    return next(e);
  });
  for (const c of COMMANDS) {
    on("command.run", { command: c.name }, async ($, e) => command($, e.command, e.args));
  }
  on("ui.render", { component: "Pane" }, async ($, e, next) => dRenderPane($, e, next));
  on("ui.render", { component: "AbovePrompt" }, async ($, e, next) => dRenderBand($, e, next));
  on("ui.close", async ($, e, next) => {
    if (e.id === PANE) isOpen = false;
    return next(e);
  });
}
export {
  register
};
