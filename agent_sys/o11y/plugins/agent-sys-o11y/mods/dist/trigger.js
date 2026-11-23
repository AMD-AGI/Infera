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
  facts: (tab) => `facts/${tab}.json`,
  triggerState: "state/trigger.json",
  active: (sid) => `state/active/${sid}.json`
};
var LEDGER_TOOL = "mcp__plugin_agent-sys-o11y_o11y__ledger_append";
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
function configDirOf(env) {
  const c = env.CLAUDE_CONFIG_DIR;
  return c ? trimSlash(c) : `${trimSlash(env.HOME || "/tmp")}/.claude`;
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
function newId(prefix) {
  const t = (/* @__PURE__ */ new Date()).toISOString().replace(/[-:.]/g, "").replace("T", "-").slice(0, 15);
  return `${prefix}-${t}-${Math.random().toString(36).slice(2, 8)}`;
}
function sanitize(value, max) {
  const s = String(value ?? "").replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g, " ").trim();
  return s.length > max ? s.slice(0, max - 1) + "\u2026" : s;
}
function typeOf(x) {
  if (x === null) return "null";
  if (Array.isArray(x)) return "array";
  if (typeof x === "number") return Number.isInteger(x) ? "integer" : "number";
  return typeof x;
}
function resolve(root, node) {
  if (node && typeof node.$ref === "string" && node.$ref.startsWith("#/$defs/")) {
    return resolve(root, root.$defs[node.$ref.slice(8)]);
  }
  return node;
}
function validate(schema, value, root = schema, at = "") {
  const s = resolve(root, schema);
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
      if (ps) out.push(...validate(ps, v, root, `${at}/${k}`));
      else if (s.additionalProperties === false) out.push(`${at}/${k}: unknown key`);
    }
  }
  if (t === "array") {
    const arr = value;
    if (typeof s.minItems === "number" && arr.length < s.minItems) out.push(`${where}: needs ${s.minItems}+ items`);
    if (s.items) arr.forEach((v, i) => out.push(...validate(s.items, v, root, `${at}/${i}`)));
  }
  return out;
}
function withDefaults(schema, value, root = schema) {
  const s = resolve(root, schema);
  if (!s || s.type !== "object" || !s.properties) return value;
  const out = { ...value ?? {} };
  for (const [k, ps0] of Object.entries(s.properties)) {
    const ps = resolve(root, ps0);
    if (out[k] === void 0 && ps.default !== void 0) out[k] = JSON.parse(JSON.stringify(ps.default));
    if (ps.type === "object" && ps.properties) out[k] = withDefaults(ps, out[k], root);
  }
  return out;
}
async function readDoc($, path, schemaName) {
  try {
    if (!await $.fs.exists(path)) return { value: null, problem: null, mtimeMs: 0 };
    const st2 = await $.fs.stat(path);
    const value = JSON.parse(await $.fs.read(path));
    const probs = schemaName ? validate(SCHEMAS[schemaName], value) : [];
    return { value, problem: probs.length ? `${path}: ${probs[0]}` : null, mtimeMs: st2.mtimeMs };
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
  const st2 = await $.fs.stat(path);
  const offset = Math.max(0, st2.size - maxBytes);
  const text = offset > 0 ? await $.fs.read(path, { offset, length: st2.size - offset }) : await $.fs.read(path);
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
var DETACH_SCRIPT = 'log=$1; shift; setsid -f "$@" >>"$log" 2>&1 </dev/null';
function detachArgv(log, argv) {
  return ["sh", "-c", DETACH_SCRIPT, "sh", log, ...argv];
}

// plugins/agent-sys-o11y/mods/src/trigger/index.ts
var SKILL_RE = /^(infera-agent-sys|agent-sys-[\w-]+)$/;
var MARK = { kick: "[o11y automated kick] ", packup: "[o11y automated packup] ", question: "[o11y automated question] " };
var sessionId = "";
var myLineage = [];
var vetoed = false;
var active = false;
var ctx = null;
var cfg = withDefaults(SCHEMAS["config.trigger"], { v: 1 });
var cfgMtime = -1;
var st = null;
var pendingIsLive = false;
var ticker = null;
var tickMs = 0;
var sidWatch = null;
var lastSpawn = {};
var lastRegistry = "";
var collectorUp = null;
var writeChain = Promise.resolve();
function abs(rel) {
  return `${ctx.fleet}/${rel}`;
}
function writer() {
  return { session_id: sessionId, role: "main", source: "detected" };
}
function serial(fn) {
  writeChain = writeChain.then(fn).catch(() => {
  });
  return writeChain;
}
function skillName(raw) {
  if (typeof raw !== "string") return null;
  const bare = raw.replace(/^\//, "").split(/\s/)[0].split(":").pop() ?? "";
  return SKILL_RE.test(bare) ? bare : null;
}
async function loadConfig($) {
  const r = await readDoc($, abs(P.config("trigger")), "config.trigger");
  if (r.mtimeMs === cfgMtime) return;
  cfgMtime = r.mtimeMs;
  if (r.problem) $.ui.log(`o11y: trigger config ignored: ${r.problem}`, { to: "debug" });
  const before = cfg.kick.interval_s;
  cfg = withDefaults(SCHEMAS["config.trigger"], r.problem ? { v: 1 } : r.value ?? { v: 1 });
  if (st && cfg.kick.interval_s !== before) st.next_kick_at = Math.min(st.next_kick_at, Date.now() + cfg.kick.interval_s * 1e3);
}
async function loadState(now, $) {
  const s = (await readDoc($, abs(P.triggerState))).value;
  if (s && s.v === 1 && s.session_id === sessionId) {
    st = { pending: null, retry: null, asked: [], ...s };
    pendingIsLive = false;
    return;
  }
  st = {
    v: 1,
    session_id: sessionId,
    seq: s?.seq ?? 0,
    // trigger records never reuse a number
    next_kick_at: now + cfg.kick.interval_s * 1e3,
    next_slow_at: now + cfg.log.slow_interval_s * 1e3,
    kicks: 0,
    kicks_since_packup: 0,
    last_packup_at: now,
    tool_calls: 0,
    turns: 0,
    pending: null,
    retry: null,
    asked: s?.asked ?? []
  };
}
function saveState($) {
  const snapshot = JSON.parse(JSON.stringify(st));
  return serial(() => writeDoc($, abs(P.triggerState), snapshot));
}
function fire($, kind, outcome, extra = {}) {
  const seq = ++st.seq;
  const rec = {
    v: 1,
    id: newId("trig"),
    ts: nowIso(),
    seq,
    kind,
    outcome,
    session_id: sessionId,
    ...extra.detail ? { detail: extra.detail.slice(0, 500) } : {},
    ...extra.text ? { text: extra.text } : {},
    ...extra.subject ? { subject_session_id: extra.subject } : {},
    counters: { tool_calls: st.tool_calls, turns: st.turns, kicks: st.kicks },
    writer: writer()
  };
  return serial(() => writeDoc($, abs(`${P.triggers}/${String(seq).padStart(8, "0")}.json`), rec));
}
function procStartOf(stat) {
  return stat.slice(stat.lastIndexOf(")") + 2).split(" ")[19] ?? null;
}
async function aliveOf($, pid, procStart) {
  try {
    const s = await $.fs.read(`/proc/${pid}/stat`);
    return procStart == null || procStartOf(s) === String(procStart);
  } catch {
    return false;
  }
}
async function tmuxServerOf($, pid) {
  try {
    const env = await $.fs.read(`/proc/${pid}/environ`);
    const kv = env.split("\0").find((x) => x.startsWith("TMUX="));
    return kv ? kv.slice(5).split(",")[0] : null;
  } catch {
    return null;
  }
}
function statusOf(s) {
  return ["busy", "idle", "waiting", "shell"].includes(s) ? s : "unknown";
}
function tmuxMatches(recTmux, target) {
  if (!recTmux || !target) return false;
  const [sess, rest] = recTmux.split(":");
  return target === sess || target === recTmux || (rest ?? "").endsWith("." + target);
}
async function refreshRegistry($) {
  const dir = `${ctx.configDir}/sessions`;
  let names = [];
  try {
    names = (await $.fs.list(dir)).map((x) => x.name).filter((n) => /^\d+\.json$/.test(n));
  } catch {
  }
  const launches = (await readJsonl($, abs(P.launches))).filter((l) => l && l.v === 1);
  const prev = (await readDoc($, abs(P.sessions))).value;
  const prevByPid = new Map((prev?.sessions ?? []).map((s) => [s.pid, s]));
  const records = [];
  for (const n of names) {
    const pid = Number(n.slice(0, -5));
    try {
      const r = JSON.parse(await $.fs.read(`${dir}/${n}`));
      if (r && typeof r.pid === "number" && typeof r.sessionId === "string") records.push(r);
      else throw new Error("shape");
    } catch {
      const p = prevByPid.get(pid);
      if (p) records.push({ __stale: p });
    }
  }
  const isMe = (r) => r.sessionId === sessionId || ctx.pid != null && r.pid === ctx.pid;
  const me = records.find(isMe);
  const myServer = ctx.pid ? await tmuxServerOf($, ctx.pid) : null;
  const out = [];
  const candidates = [];
  for (const r of records) {
    if (r.__stale) {
      out.push({ ...r.__stale, last_good: true });
      continue;
    }
    let role = "other";
    let parent = null;
    let purpose = null;
    if (isMe(r)) role = "main";
    else {
      const l = launches.findLast(
        (x) => x.session_id === r.sessionId || x.name && x.name === r.name || tmuxMatches(r.tmux ?? null, x.tmux_target)
      );
      if (l && l.role === "child") {
        role = "child";
        parent = l.parent_session_id ?? sessionId;
        purpose = l.purpose ?? null;
      } else if (!l && me && r.tmux && typeof r.cwd === "string" && (r.cwd === ctx.launchDir || r.cwd.startsWith(ctx.launchDir + "/")) && (r.startedAt ?? 0) > (me.startedAt ?? 0)) {
        const server = await tmuxServerOf($, r.pid);
        if (!myServer || !server || server === myServer) {
          if (await aliveOf($, r.pid, r.procStart ?? null)) candidates.push({ sid: r.sessionId, name: r.name ?? "", tmux: r.tmux, cwd: r.cwd });
        }
      }
    }
    if (role === "other") continue;
    const alive = await aliveOf($, r.pid, r.procStart ?? null);
    const p = prevByPid.get(r.pid);
    const lineage = role === "main" ? [...myLineage] : p ? [...p.lineage ?? []] : [];
    if (p && p.session_id !== r.sessionId && p.proc_start === (r.procStart ?? null) && !lineage.includes(p.session_id)) {
      lineage.push(p.session_id);
    }
    const tp = r.cwd ? `${ctx.configDir}/projects/${await slug(String(r.cwd))}/${r.sessionId}.jsonl` : null;
    out.push({
      session_id: role === "main" ? sessionId : r.sessionId,
      pid: r.pid,
      proc_start: r.procStart ?? null,
      alive,
      crashed: !alive,
      name: r.name ?? null,
      cwd: r.cwd ?? null,
      tmux: r.tmux ?? null,
      transcript_path: tp && await $.fs.exists(tp) ? tp : null,
      status: statusOf(r.status),
      waiting_for: r.waitingFor ?? null,
      ...r.statusUpdatedAt ? { status_since: nowIso(r.statusUpdatedAt) } : {},
      role,
      parent_session_id: parent,
      lineage,
      ...r.startedAt ? { started_at: nowIso(r.startedAt) } : {},
      ...r.updatedAt ? { updated_at: nowIso(r.updatedAt) } : {},
      version: r.version ?? null,
      kind: r.kind ?? null,
      purpose,
      active: role === "main",
      fleet_dir: ctx.fleet,
      config_dir: ctx.configDir,
      launch_dir: role === "main" ? ctx.launchDir : null
    });
  }
  const body = JSON.stringify(out);
  if (body !== lastRegistry) {
    lastRegistry = body;
    const doc = { v: 1, generated_at: nowIso(), main_session_id: sessionId, sessions: out };
    await serial(() => writeDoc($, abs(P.sessions), doc));
  }
  return { sessions: out, candidates };
}
function render(template, vars) {
  return template.replace(/\{(\w+)\}/g, (m, k) => k in vars ? vars[k] : m);
}
function childrenText(sessions) {
  const max = cfg.kick.max_value_chars;
  const kids = sessions.filter((s) => s.role === "child" && s.alive);
  if (!kids.length) return "none";
  return kids.map((s) => `${sanitize(s.name ?? s.session_id, max)}${s.tmux ? ` (tmux ${sanitize(s.tmux.split(":")[0], max)})` : ""}`).join(", ");
}
function isPending(now) {
  const p = st.pending;
  if (!p) return false;
  if (pendingIsLive) return true;
  return now - p.submitted_at < 3 * cfg.kick.interval_s * 1e3;
}
function submit($, kind, text, about) {
  st.pending = { kind, submitted_at: Date.now(), ...about ? { about } : {} };
  pendingIsLive = true;
  $.prompt.submit({ text }).then(
    (r) => {
      st.pending = null;
      if (r && r.drop) {
        st.retry = { kind, text, ...about ? { about } : {} };
        void fire($, kind, "refused", { detail: String(r.drop) });
      }
      void saveState($);
    },
    (err) => {
      st.pending = null;
      void fire($, kind, "failed", { detail: String(err?.message ?? err) });
      void saveState($);
    }
  );
}
async function scheduleKicks($, now, sessions) {
  if (!cfg.enabled || !cfg.kick.enabled || now < st.next_kick_at) return;
  const step = cfg.kick.interval_s * 1e3;
  while (st.next_kick_at <= now) st.next_kick_at += step;
  const usePackup = cfg.packup.enabled && (st.kicks_since_packup + 1 >= cfg.packup.every_n_kicks || now - st.last_packup_at >= cfg.packup.interval_s * 1e3);
  const kind = usePackup ? "packup" : "kick";
  if (isPending(now)) {
    await fire($, kind, "skipped_pending", { detail: `${st.pending.kind} pending since ${nowIso(st.pending.submitted_at)}` });
    return;
  }
  const max = cfg.kick.max_value_chars;
  const book = cfg.task_book.startsWith("/") ? cfg.task_book : `${ctx.launchDir}/${cfg.task_book}`;
  const vars = { task_book: sanitize(book, max), children: childrenText(sessions), ledger_tool: LEDGER_TOOL };
  const text = MARK[kind] + render(usePackup ? cfg.packup.template : cfg.kick.template, vars);
  st.kicks++;
  if (usePackup) {
    st.kicks_since_packup = 0;
    st.last_packup_at = now;
  } else st.kicks_since_packup++;
  submit($, kind, text);
  await fire($, kind, "submitted", { text });
}
async function askAboutUnregistered($, now, candidates) {
  if (!cfg.question.enabled || !candidates.length || isPending(now)) return;
  const c = candidates.find((x) => !st.asked.includes(x.sid));
  if (!c) return;
  const max = cfg.kick.max_value_chars;
  const vars = { name: sanitize(c.name || c.sid, max), tmux: sanitize(c.tmux, max), cwd: sanitize(c.cwd, max) };
  const text = MARK.question + render(cfg.question.template, vars);
  st.asked.push(c.sid);
  submit($, "question", text, c.sid);
  await fire($, "question", "submitted", { text, subject: c.sid });
}
async function enqueue($, kind, reason) {
  if (!ctx || !st) return;
  const id = newId(kind);
  const item = { v: 1, id, ts: nowIso(), kind, reason, session_id: sessionId, counters: { tool_calls: st.tool_calls, turns: st.turns }, writer: writer() };
  await serial(() => writeDoc($, abs(`${P.queue}/${id}.json`), item));
  const fk = kind === "display" ? "refresh" : "log";
  await fire($, fk, "enqueued", { detail: reason });
  const now = Date.now();
  if (now - (lastSpawn[kind] ?? 0) < 5e3) return;
  lastSpawn[kind] = now;
  const argv = [`${ctx.pluginRoot}/server/py.sh`, "-m", "o11y_analyze", "--kind", kind];
  try {
    const r = await $.process.run(detachArgv(abs("state/analyze.spawn.log"), argv), {
      env: { AGENT_SYS_O11Y_FLEET_DIR: ctx.fleet, CLAUDE_CONFIG_DIR: ctx.configDir, AGENT_SYS_HOME: ctx.prefix },
      timeoutMs: 1e4
    });
    if (r.exitCode !== 0) await fire($, fk, "failed", { detail: r.stderr.slice(0, 300) });
  } catch (err) {
    await fire($, fk, "failed", { detail: err.message });
  }
}
function later($, kind, reason) {
  $.clock.after(1, () => enqueue($, kind, reason));
}
async function detach($, argv) {
  try {
    await $.process.run(detachArgv(abs("state/spawn.log"), argv), {
      env: { AGENT_SYS_O11Y_FLEET_DIR: ctx.fleet, CLAUDE_CONFIG_DIR: ctx.configDir, AGENT_SYS_HOME: ctx.prefix },
      timeoutMs: 1e4
    });
  } catch {
  }
}
async function checkCollector($) {
  const url = await $.env.get("AGENT_SYS_O11Y_OTEL_HEALTH_URL");
  if (!url) return;
  let up = false;
  try {
    up = (await $.http.fetch(url)).ok === true;
  } catch {
  }
  if (up === false && collectorUp !== false) {
    await detach($, [
      `${ctx.pluginRoot}/server/py.sh`,
      "-m",
      "o11y_core",
      "--session-id",
      sessionId,
      "--role",
      "main",
      "event",
      "report",
      "--severity",
      "ERROR",
      "--name",
      "o11y.otel.collector_down",
      "--body",
      `collector at ${sanitize(url, 200)} is not healthy`,
      "--source",
      "detector"
    ]);
  }
  collectorUp = up;
}
async function tick($) {
  if (!ctx || !st) return;
  try {
    await watchSessionId($);
    const now = Date.now();
    await loadConfig($);
    const { sessions, candidates } = await refreshRegistry($);
    await scheduleKicks($, now, sessions);
    await askAboutUnregistered($, now, candidates);
    if (cfg.enabled && now >= st.next_slow_at) {
      while (st.next_slow_at <= now) st.next_slow_at += cfg.log.slow_interval_s * 1e3;
      await enqueue($, "log", "timer");
    }
    await checkCollector($);
    await saveState($);
    if (cfg.registry_interval_s * 1e3 !== tickMs) startTicker($);
  } catch (err) {
    $.ui.log(`o11y: trigger tick: ${err.message}`, { to: "debug" });
  }
}
function startTicker($) {
  if (ticker) ticker.cancel();
  tickMs = cfg.registry_interval_s * 1e3;
  ticker = $.clock.every(tickMs, () => tick($));
}
async function selfPid($) {
  try {
    return Number((await $.fs.read("/proc/self/stat")).split(" ")[0]) || null;
  } catch {
    return null;
  }
}
async function makeCtx($) {
  const env = await readEnv($);
  const launchDir = String(await $.session.root());
  return {
    fleet: await fleetDirOf(env, launchDir),
    launchDir,
    configDir: configDirOf(env),
    prefix: prefixOf(env),
    pluginRoot: $.plugin.root,
    pid: await selfPid($)
  };
}
async function activate($, by, lineage = []) {
  if (active || vetoed) return;
  active = true;
  myLineage = lineage;
  ctx ??= await makeCtx($);
  await writeDoc($, abs(P.active(sessionId)), { v: 1, session_id: sessionId, by, at: nowIso(), lineage });
  await loadConfig($);
  await loadState(Date.now(), $);
  startTicker($);
  await tick($);
  $.ui.log(`o11y active (${by}); fleet ${ctx.fleet}`);
}
async function watchSessionId($) {
  const sid = await $.session.id();
  if (!sid || sid === sessionId) return;
  const old = sessionId;
  sessionId = sid;
  if (!active || !ctx) return;
  const prev = (await readDoc($, abs(P.active(old)))).value;
  myLineage = [...prev?.lineage ?? myLineage, old];
  await writeDoc($, abs(P.active(sid)), { v: 1, session_id: sid, by: `cleared from ${old}`, at: nowIso(), lineage: myLineage });
  st = { ...st, session_id: sid };
  await tick($);
}
function register(on) {
  on("session.start", async ($, e, next) => {
    sessionId = await $.session.id();
    const env = await $.env.get("AGENT_SYS_O11Y");
    vetoed = env !== void 0 && env !== "true";
    if (sidWatch) sidWatch.cancel();
    sidWatch = $.clock.every(5e3, () => watchSessionId($));
    if (!vetoed) {
      ctx = await makeCtx($);
      const armed = (await readDoc($, abs(P.active(sessionId)))).value;
      if (env === "true") await activate($, "env");
      else if (armed) await activate($, `re-armed (${armed.by})`, armed.lineage ?? []);
    }
    return next(e);
  });
  on("tool.call", async ($, e, next) => {
    const r = await next(e);
    if (e.agentId || vetoed) return r;
    if (!active) {
      if (e.tool === "Skill") {
        const s = skillName(e.skill);
        if (s) $.clock.after(1, () => activate($, `skill ${s}`));
      }
      return r;
    }
    st.tool_calls++;
    if (st.tool_calls % cfg.refresh.every_n_tool_calls === 0) later($, "display", "tool_calls");
    if (st.tool_calls % cfg.log.every_n_tool_calls === 0) later($, "log", "tool_calls");
    return r;
  });
  on("command.run", async ($, e, next) => {
    if (!active && !vetoed && e.origin?.kind === "composer") {
      const s = skillName(e.command);
      if (s) $.clock.after(1, () => activate($, `skill /${s}`));
    }
    return next(e);
  });
  on("turn.start", async ($, e, next) => {
    if (active && st?.pending && !pendingIsLive && !e.agentId) st.pending = null;
    return next(e);
  });
  on("turn.complete", async ($, e, next) => {
    if (!active || !st || e.agentId) return next(e);
    st.turns++;
    if (st.retry && !e.isAborted) {
      const r = st.retry;
      st.retry = null;
      $.clock.after(1, async () => {
        submit($, r.kind, r.text, r.about);
        await fire($, r.kind, "submitted", { text: r.text, detail: "retry after refusal", ...r.about ? { subject: r.about } : {} });
      });
    }
    if (st.turns % cfg.refresh.every_n_turns === 0) later($, "display", "turns");
    if (st.turns % cfg.log.every_n_turns === 0) later($, "log", "turns");
    return next(e);
  });
  on("session.end", async ($, e, next) => {
    if (active && st) {
      void saveState($);
      if (e.reason !== "clear" && e.reason !== "resume") {
        if (ticker) ticker.cancel();
        ticker = null;
        const launcher = `${ctx.pluginRoot}/bin/o11y-launch`;
        if (await $.fs.exists(launcher)) void detach($, [launcher, "--sweep"]);
      }
    }
    return next(e);
  });
}
export {
  register
};
