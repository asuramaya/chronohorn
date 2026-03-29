use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

pub const PARAMETER_GOLF_MAGIC: i32 = 20240520;
pub const PARAMETER_GOLF_VERSION: i32 = 1;
pub const HEADER_INTS: usize = 256;
pub const HEADER_BYTES: usize = HEADER_INTS * std::mem::size_of::<i32>();

#[derive(Debug, Clone, Serialize)]
pub struct DataRootReport {
    pub requested_path: String,
    pub is_symlink: bool,
    pub symlink_target: Option<String>,
    pub exists: bool,
    pub resolved_path: Option<String>,
    pub kind: String,
    pub train_shard_count: usize,
    pub val_shard_count: usize,
}

pub fn inspect_data_root(root: &Path) -> DataRootReport {
    let requested_path = root.display().to_string();
    let symlink_meta = fs::symlink_metadata(root).ok();
    let is_symlink = symlink_meta
        .as_ref()
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false);
    let symlink_target = if is_symlink {
        fs::read_link(root)
            .ok()
            .map(|target| target.display().to_string())
    } else {
        None
    };
    let exists = root.exists();
    let resolved_path = if exists {
        fs::canonicalize(root)
            .ok()
            .map(|path| path.display().to_string())
    } else {
        None
    };
    let train_shard_count = if exists {
        count_shards(root, "fineweb_train_")
    } else {
        0
    };
    let val_shard_count = if exists {
        count_shards(root, "fineweb_val_")
    } else {
        0
    };
    let path_label = requested_path.to_ascii_lowercase();
    let kind = if !exists {
        "missing".to_string()
    } else if train_shard_count > 0 && val_shard_count > 0 {
        if path_label.contains("replay_root") {
            "replay_root".to_string()
        } else if path_label.contains("local_code_tokens") {
            "local_code_tokens".to_string()
        } else {
            "shard_root".to_string()
        }
    } else {
        "unknown".to_string()
    };
    DataRootReport {
        requested_path,
        is_symlink,
        symlink_target,
        exists,
        resolved_path,
        kind,
        train_shard_count,
        val_shard_count,
    }
}

fn count_shards(root: &Path, prefix: &str) -> usize {
    fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok().map(|row| row.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with(prefix) && name.ends_with(".bin"))
                .unwrap_or(false)
        })
        .count()
}

pub fn load_shard(path: &Path) -> Result<Vec<usize>, String> {
    let blob = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    if blob.len() >= HEADER_BYTES {
        let mut header = Vec::with_capacity(HEADER_INTS);
        for chunk in blob[..HEADER_BYTES].chunks_exact(4) {
            header.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if header.len() >= 3
            && header[0] == PARAMETER_GOLF_MAGIC
            && header[1] == PARAMETER_GOLF_VERSION
        {
            let token_count =
                usize::try_from(header[2]).map_err(|_| "negative token count".to_string())?;
            let payload = &blob[HEADER_BYTES..];
            let mut out = Vec::with_capacity(token_count);
            for chunk in payload.chunks_exact(2).take(token_count) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as usize);
            }
            if out.len() == token_count {
                return Ok(out);
            }
        }
    }
    let mut out = Vec::with_capacity(blob.len() / 2);
    for chunk in blob.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as usize);
    }
    Ok(out)
}

pub fn list_shards(root: &Path, prefix: &str) -> Result<Vec<PathBuf>, String> {
    let mut rows: Vec<PathBuf> = fs::read_dir(root)
        .map_err(|err| format!("read_dir {}: {err}", root.display()))?
        .filter_map(|entry| entry.ok().map(|row| row.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with(prefix) && name.ends_with(".bin"))
                .unwrap_or(false)
        })
        .collect();
    rows.sort();
    if rows.is_empty() {
        return Err(format!(
            "no shards found under {} with prefix {prefix}",
            root.display()
        ));
    }
    Ok(rows)
}

pub fn take_train_tokens(root: &Path, token_budget: usize) -> Result<Vec<usize>, String> {
    let shards = list_shards(root, "fineweb_train_")?;
    let mut out = Vec::with_capacity(token_budget);
    for shard in shards {
        if out.len() >= token_budget {
            break;
        }
        let tokens = load_shard(&shard)?;
        let left = token_budget - out.len();
        out.extend(tokens.into_iter().take(left));
    }
    if out.len() < 4 {
        return Err("need at least 4 train tokens".to_string());
    }
    Ok(out)
}

pub fn take_val_tokens(root: &Path, token_budget: usize) -> Result<Vec<usize>, String> {
    let shards = list_shards(root, "fineweb_val_")?;
    let mut out = Vec::with_capacity(token_budget);
    for shard in shards {
        if out.len() >= token_budget {
            break;
        }
        let tokens = load_shard(&shard)?;
        let left = token_budget - out.len();
        out.extend(tokens.into_iter().take(left));
    }
    if out.is_empty() {
        return Err("no validation tokens loaded".to_string());
    }
    Ok(out)
}
