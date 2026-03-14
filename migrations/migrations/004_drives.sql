-- Migration: 004_drives
-- Creates the drives table for the affect/motivational drive system.
-- Drive rows persist behavioral weight biases across sessions.
-- value: 0.0-1.0 current drive strength
-- baseline: 0.0-1.0 equilibrium value (decays toward this)
-- decay_rate: per-reflection-cycle decay fraction toward baseline (0.0-1.0)
-- Run: python migrations/apply.py 004_drives.sql

CREATE TABLE IF NOT EXISTS `{prefix}drives` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `name`        VARCHAR(64)   NOT NULL COMMENT 'Slug identifier, e.g. curiosity, task-completion',
    `description` TEXT          NOT NULL COMMENT 'What this drive promotes in behavior',
    `value`       FLOAT         NOT NULL DEFAULT 0.5 COMMENT 'Current strength 0.0-1.0',
    `baseline`    FLOAT         NOT NULL DEFAULT 0.5 COMMENT 'Equilibrium value 0.0-1.0',
    `decay_rate`  FLOAT         NOT NULL DEFAULT 0.05 COMMENT 'Fraction to decay toward baseline per reflection cycle',
    `source`      ENUM('system','user','reflection') NOT NULL DEFAULT 'system',
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
