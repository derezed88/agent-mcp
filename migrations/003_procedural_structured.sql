-- Migration: 003_procedural_structured
-- Adds structured columns to {prefix}procedural for task_type, steps, outcome, run tracking.
-- Uses ALTER TABLE to preserve existing row IDs (Qdrant point alignment).
-- Run: python migrations/apply.py 003

ALTER TABLE `{prefix}procedural`
    ADD COLUMN `task_type`     VARCHAR(120)  NOT NULL DEFAULT '' AFTER `topic`,
    ADD COLUMN `steps`         MEDIUMTEXT    DEFAULT NULL COMMENT 'JSON array: [{"step":N,"action":"...","tool":"...","note":"..."}]',
    ADD COLUMN `outcome`       ENUM('unknown','success','partial','failure') NOT NULL DEFAULT 'unknown' AFTER `steps`,
    ADD COLUMN `run_count`     SMALLINT      NOT NULL DEFAULT 1 COMMENT 'Total times this procedure has been executed',
    ADD COLUMN `success_count` SMALLINT      NOT NULL DEFAULT 0 COMMENT 'Count of successful runs',
    ADD COLUMN `notes`         TEXT          DEFAULT NULL COMMENT 'Free-form lessons learned, caveats, edge cases',
    ADD COLUMN `last_run_at`   TIMESTAMP     DEFAULT NULL COMMENT 'Timestamp of most recent execution',
    ADD KEY `idx_task_type` (`task_type`),
    ADD KEY `idx_outcome` (`outcome`);
