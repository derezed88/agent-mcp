-- Migration: 009_cognition
-- Dedicated table for cognitive loop outputs (reflection, goal_health,
-- self-model, prospective reminders, tool logs/failures).
-- These were previously scattered across samaritan_memory_shortterm/longterm
-- using topic-prefix conventions, making them vulnerable to !memreview renames
-- and polluting the conversation memory tables.
-- Run via: python migrations/apply.py 009_cognition.sql

DROP TABLE IF EXISTS `{prefix}cognition`;

CREATE TABLE `{prefix}cognition` (
    `id`            INT(11)      NOT NULL AUTO_INCREMENT,
    `origin`        ENUM(
                        'reflection',
                        'goal_health',
                        'self_model',
                        'prospective',
                        'tool_log',
                        'tool_failure',
                        'summary'
                    ) NOT NULL,
    `topic`         VARCHAR(255) NOT NULL DEFAULT '',
    `content`       TEXT         NOT NULL,
    `importance`    TINYINT      NOT NULL DEFAULT 5,
    `source`        ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'session',
    `session_id`    VARCHAR(255) NOT NULL DEFAULT '',
    `last_accessed` TIMESTAMP    NULL DEFAULT NULL,
    `created_at`    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_origin`          (`origin`),
    KEY `idx_origin_topic`    (`origin`, `topic`),
    KEY `idx_importance`      (`importance`),
    KEY `idx_created`         (`created_at`),
    KEY `idx_last_accessed`   (`last_accessed`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
