-- Migration: 002_memory_type_column
-- Adds a `type` column to shortterm and longterm memory tables.
-- Existing rows default to 'context' (unclassified).
-- Run via: python migrations/apply.py 002_memory_type_column.sql

ALTER TABLE `{prefix}memory_shortterm`
    ADD COLUMN `type` ENUM(
        'context',
        'goal',
        'plan',
        'belief',
        'episodic',
        'semantic',
        'procedural',
        'autobiographical',
        'prospective',
        'conditioned'
    ) NOT NULL DEFAULT 'context' AFTER `source`,
    ADD KEY `idx_type` (`type`);

ALTER TABLE `{prefix}memory_longterm`
    ADD COLUMN `type` ENUM(
        'context',
        'goal',
        'plan',
        'belief',
        'episodic',
        'semantic',
        'procedural',
        'autobiographical',
        'prospective',
        'conditioned'
    ) NOT NULL DEFAULT 'context' AFTER `source`,
    ADD KEY `idx_type` (`type`);
