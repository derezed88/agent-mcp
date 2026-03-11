-- Migration: 001_memory_types
-- Creates typed memory tables for all three databases.
-- Each database uses its own prefix: samaritan_, qwen_, test_
-- Run once per database: mysql <dbname> < 001_memory_types.sql
-- Or use the apply script: python migrations/apply.py 001_memory_types.sql

-- ============================================================
-- GOALS
-- Active objectives. Never age out below importance 9.
-- childof / parentof: JSON arrays of goal IDs in same table.
-- memory_link: JSON array of shortterm/longterm memory row IDs.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}goals` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `title`       VARCHAR(255)  NOT NULL,
    `description` TEXT          NOT NULL,
    `status`      ENUM('active','done','blocked','abandoned') NOT NULL DEFAULT 'active',
    `importance`  TINYINT(4)    NOT NULL DEFAULT 9,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `childof`     TEXT          DEFAULT NULL COMMENT 'JSON array of parent goal IDs',
    `parentof`    TEXT          DEFAULT NULL COMMENT 'JSON array of child goal IDs',
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_status` (`status`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- PLANS
-- Ordered steps toward a goal. One row per step.
-- goal_id: FK into goals table (same prefix).
-- step_order: ascending integer for ordering steps.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}plans` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `goal_id`     INT(11)       NOT NULL,
    `step_order`  INT(11)       NOT NULL DEFAULT 1,
    `description` TEXT          NOT NULL,
    `status`      ENUM('pending','in_progress','done','skipped') NOT NULL DEFAULT 'pending',
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_goal` (`goal_id`),
    KEY `idx_goal_order` (`goal_id`, `step_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- BELIEFS
-- Asserted world-state facts.
-- confidence: 1-10 scale (similar to importance).
-- status: active or retracted (soft-delete).
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}beliefs` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `confidence`  TINYINT(4)    NOT NULL DEFAULT 7,
    `status`      ENUM('active','retracted') NOT NULL DEFAULT 'active',
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- EPISODIC MEMORY
-- Recollection of specific experiences, events, situations.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}episodic` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- SEMANTIC MEMORY
-- Knowledge of facts, concepts, ideas, general world knowledge.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}semantic` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- PROCEDURAL MEMORY
-- Memory for skills, habits, and task steps.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}procedural` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- AUTOBIOGRAPHICAL MEMORY
-- Blend of episodic and semantic that defines system/person identity and story.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}autobiographical` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 7,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- PROSPECTIVE MEMORY
-- Remembering to perform a planned action or intention in the future.
-- due_at: optional timestamp or free-text description of when.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}prospective` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `due_at`      VARCHAR(255)  DEFAULT NULL COMMENT 'Timestamp or free-text future trigger (e.g. "next Monday", "2026-04-01 09:00")',
    `status`      ENUM('pending','done','missed') NOT NULL DEFAULT 'pending',
    `importance`  TINYINT(4)    NOT NULL DEFAULT 7,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_status` (`status`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- CONDITIONED MEMORY
-- Learned reactions to specific triggers (classical conditioning model).
-- trigger: the stimulus pattern (text, keyword, or structured condition).
-- reaction: the learned response behavior or action.
-- strength: 1-10, how strongly the conditioning is reinforced.
-- ============================================================

CREATE TABLE IF NOT EXISTS `{prefix}conditioned` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `trigger`     TEXT          NOT NULL COMMENT 'Stimulus pattern or condition',
    `reaction`    TEXT          NOT NULL COMMENT 'Learned response or behavior',
    `strength`    TINYINT(4)    NOT NULL DEFAULT 5 COMMENT '1-10 reinforcement strength',
    `status`      ENUM('active','extinguished') NOT NULL DEFAULT 'active',
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL COMMENT 'JSON array of memory row IDs',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
