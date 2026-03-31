pub(crate) fn print_usage() {
    println!("chronohorn");
    println!();
    println!("Core:");
    println!("  chronohorn inspect-npz <checkpoint.npz>");
    println!("  chronohorn inspect-data-root [@alias|path]");
    println!("  chronohorn print-data-home");
    println!("  chronohorn print-parallel-runtime");
    println!("  chronohorn doctrine");
    println!("  chronohorn doctrine-json");
    println!("  chronohorn design");
    println!();
    println!("Promoted causal-bank runtime:");
    println!(
        "  chronohorn run-causal-bank-checkpoint <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
    );
    println!(
        "  chronohorn audit-causal-bank-checkpoint <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-causal-bank-exact-checkpoint <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens]"
    );
    println!(
        "  chronohorn audit-causal-bank-exact-checkpoint <checkpoint-path|bundle-dir> <summary.json> <data-root> [val_tokens] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn build-causal-bank-ngram-oracle-budgeted-table <data-root> <artifact-path> [train_tokens] [report_every] [profile=tiny|medium|absurd] [oracle_stride]"
    );
    println!(
        "  chronohorn build-causal-bank-ngram-oracle-row-stats <data-root> <artifact-path> [train_tokens] [report_every] [profile=tiny|medium|absurd] [oracle_stride]"
    );
    println!(
        "  chronohorn pack-causal-bank-ngram-oracle-row-stats <stats-artifact> <packed-artifact> [target_bytes=8388608]"
    );
    println!(
        "  chronohorn run-causal-bank-ngram-bulk-from-table <checkpoint-path|bundle-dir> <summary.json> <data-root> <artifact-path> [val_tokens] [report_every]"
    );
    println!();
    println!("Additional checkpoint and table probes remain available for the");
    println!("current implementation line, but they are not the promoted public surface.");
    println!("Legacy internal command names remain accepted for compatibility.");
    println!("Use `chronohorn help-archive` for older bridge-family commands.");
    println!();
    println!("Runtime checks and doctrine:");
    println!(
        "  chronohorn audit-demo <legal|self-include|future-peek|length-peek|boundary-double-update|reported-gold-cheat>"
    );
    println!("  chronohorn audit-packed-memory <data-root> [token-budget] [trigram-buckets]");
    println!("  chronohorn compare-packed-memory <checkpoint-path|bundle-dir> <summary.json> [data-root]");
    println!("  chronohorn oracle-clean-summary <attack.json> [top-n]");
    println!();
    println!("Archive bridge families:");
    println!("  chronohorn help-archive");
    println!();
    println!("Parallel runtime:");
    println!("  Set CHRONOHORN_THREADS=<n> to override the Rayon thread pool width.");
}

pub(crate) fn print_archive_usage() {
    println!("chronohorn archive help");
    println!();
    println!("These commands remain available for older bridge families and");
    println!("implementation archaeology. They are not the promoted public surface.");
    println!();
    println!("  chronohorn train-byte-bridge [radius] [stride] [max-files]");
    println!("  chronohorn run-byte-bridge-codec [radius] [stride] [max-files]");
    println!(
        "  chronohorn run-token-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [candidate_k]"
    );
    println!(
        "  chronohorn audit-token-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [candidate_k] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-copy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-copy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-skip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [train_stride] [candidate_k]"
    );
    println!(
        "  chronohorn audit-token-skip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [train_stride] [candidate_k] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-skipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-skipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-match-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-match-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchcopy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-matchcopy-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchskip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-matchskip-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchskip-token-oracle-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [oracle_radius] [oracle_stride] [gate_target_kind] [trust_target_kind]"
    );
    println!(
        "  chronohorn audit-token-matchskip-token-oracle-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [oracle_radius] [oracle_stride] [gate_target_kind] [trust_target_kind] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-c3c7 <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [train_steps] [teacher_start_step] [oracle_radius] [oracle_stride]"
    );
    println!(
        "  chronohorn audit-token-c3c7 <data-root> [train_tokens] [trigram_buckets] [val_tokens] [train_stride] [train_steps] [teacher_start_step] [oracle_radius] [oracle_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-conker3 <data-root> [train_tokens] [val_tokens] [scale] [train_stride] [epochs] [negatives] [learning_rate]"
    );
    println!(
        "  chronohorn audit-token-conker3 <data-root> [train_tokens] [val_tokens] [scale] [train_stride] [epochs] [negatives] [learning_rate] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn analyze-token-matchskip-oracle-entropy <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [oracle_radius] [oracle_stride]"
    );
    println!(
        "  chronohorn run-token-matchskip-bundle-json <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-matchskip-manifest-json <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-experiment-matrix <data-root> [train_token_budget] [trigram_buckets] [skip_buckets] [val_token_budget] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn run-token-matchskipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp]"
    );
    println!(
        "  chronohorn audit-token-matchskipcopy-bridge <data-root> [train_tokens] [trigram_buckets] [skip_buckets] [val_tokens] [match_depth] [copy_window] [candidate_k] [train_stride] [copy_decay_bp] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-decay-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [decay_bp] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-decay-bridge <data-root> [train_tokens] [trigram_buckets] [val_tokens] [decay_bp] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-word-bridge <data-root> [train_tokens] [trigram_buckets] [word_buckets] [val_tokens] [word_window] [suffix_len] [boundary_markers] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-word-bridge <data-root> [train_tokens] [trigram_buckets] [word_buckets] [val_tokens] [word_window] [suffix_len] [boundary_markers] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
    println!(
        "  chronohorn run-token-column-bridge <data-root> [train_tokens] [trigram_buckets] [slot_buckets] [slot_period] [val_tokens] [candidate_k] [train_stride]"
    );
    println!(
        "  chronohorn audit-token-column-bridge <data-root> [train_tokens] [trigram_buckets] [slot_buckets] [slot_period] [val_tokens] [candidate_k] [train_stride] [chunk_size] [max_chunks]"
    );
}
