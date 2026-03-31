//! Generic Chronohorn runtime infrastructure.
//!
//! This crate owns backend- and family-agnostic Rust execution units that are
//! shared by promoted family runtimes and root CLIs.

pub mod audit;
pub mod bridge;
pub mod checkpoint;
pub mod data;
pub mod demo;
pub mod oracle;
pub mod protocol;
pub mod runtime;
