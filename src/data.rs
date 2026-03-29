use std::fs;
use std::path::{Path, PathBuf};

pub const PARAMETER_GOLF_MAGIC: i32 = 20240520;
pub const PARAMETER_GOLF_VERSION: i32 = 1;
pub const HEADER_INTS: usize = 256;
pub const HEADER_BYTES: usize = HEADER_INTS * std::mem::size_of::<i32>();

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
