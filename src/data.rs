use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const PARAMETER_GOLF_MAGIC: i32 = 20240520;
pub const PARAMETER_GOLF_VERSION: i32 = 1;
pub const HEADER_INTS: usize = 256;
pub const HEADER_BYTES: usize = HEADER_INTS * std::mem::size_of::<i32>();
pub const CHRONOHORN_DATA_HOME_ENV: &str = "CHRONOHORN_DATA_HOME";
pub const CHRONOHORN_ROOTS_FILE: &str = "roots.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRootAlias {
    pub alias: String,
    pub path: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DataRootRegistryFile {
    default_alias: Option<String>,
    #[serde(default)]
    aliases: Vec<DataRootAlias>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DataHomeReport {
    pub workspace_root: String,
    pub data_home: String,
    pub registry_path: String,
    pub registry_exists: bool,
    pub default_alias: String,
    pub aliases: Vec<DataRootAlias>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DataRootReport {
    pub requested_path: String,
    pub is_symlink: bool,
    pub symlink_target: Option<String>,
    pub exists: bool,
    pub resolved_path: Option<String>,
    pub kind: String,
    pub claim_tier: String,
    pub promotion_ready: bool,
    pub recommended_use: String,
    pub blocked_reason: Option<String>,
    pub train_shard_count: usize,
    pub val_shard_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedDataRoot {
    pub input: String,
    pub resolution_kind: String,
    pub alias: Option<String>,
    pub selected_path: String,
    pub data_home: String,
    pub registry_path: String,
    pub default_alias: String,
    pub report: DataRootReport,
}

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
        .to_path_buf()
}

pub fn default_data_home() -> PathBuf {
    workspace_root().join("chronohorn").join("data")
}

pub fn data_home() -> PathBuf {
    std::env::var_os(CHRONOHORN_DATA_HOME_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(default_data_home)
}

pub fn data_registry_path() -> PathBuf {
    data_home().join(CHRONOHORN_ROOTS_FILE)
}

fn builtin_aliases() -> Vec<DataRootAlias> {
    let workspace = workspace_root();
    vec![
        DataRootAlias {
            alias: "default".to_string(),
            path: "@replay".to_string(),
            note: Some("meta-alias resolved to the configured default alias".to_string()),
        },
        DataRootAlias {
            alias: "replay".to_string(),
            path: "/tmp/chronohorn_replay_root".to_string(),
            note: Some("architecture-only replay root".to_string()),
        },
        DataRootAlias {
            alias: "local-code".to_string(),
            path: "/tmp/chronohorn_local_code_tokens".to_string(),
            note: Some("architecture-only local code token root".to_string()),
        },
        DataRootAlias {
            alias: "fineweb".to_string(),
            path: workspace
                .join("conker-standalone")
                .join("conker")
                .join("data")
                .join("datasets")
                .join("fineweb10B_sp1024")
                .display()
                .to_string(),
            note: Some(
                "legacy FineWeb alias; override this in roots.json with a stored local shard root"
                    .to_string(),
            ),
        },
    ]
}

fn load_registry_file(path: &Path) -> Option<DataRootRegistryFile> {
    let blob = fs::read_to_string(path).ok()?;
    serde_json::from_str::<DataRootRegistryFile>(&blob).ok()
}

fn merged_aliases() -> (String, Vec<DataRootAlias>) {
    let registry_path = data_registry_path();
    let registry = load_registry_file(&registry_path);
    let mut aliases = builtin_aliases();
    if let Some(registry) = &registry {
        for override_row in &registry.aliases {
            if let Some(existing) = aliases
                .iter_mut()
                .find(|row| row.alias == override_row.alias)
            {
                *existing = override_row.clone();
            } else {
                aliases.push(override_row.clone());
            }
        }
    }
    let default_alias = registry
        .and_then(|row| row.default_alias)
        .unwrap_or_else(|| "replay".to_string());
    aliases.retain(|row| row.alias != "default");
    (default_alias, aliases)
}

pub fn data_home_report() -> DataHomeReport {
    let (default_alias, aliases) = merged_aliases();
    let registry_path = data_registry_path();
    DataHomeReport {
        workspace_root: workspace_root().display().to_string(),
        data_home: data_home().display().to_string(),
        registry_path: registry_path.display().to_string(),
        registry_exists: registry_path.exists(),
        default_alias,
        aliases,
    }
}

fn inspect_raw_data_root(root: &Path) -> DataRootReport {
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
    let (claim_tier, promotion_ready, recommended_use, blocked_reason) = match kind.as_str() {
        "missing" => (
            "blocked".to_string(),
            false,
            "repair_or_replace_data_root".to_string(),
            Some(if is_symlink {
                "broken_symlink".to_string()
            } else {
                "missing_path".to_string()
            }),
        ),
        "replay_root" => (
            "architecture_only".to_string(),
            false,
            "architecture_search_only".to_string(),
            Some("synthetic_replay_not_real_eval_distribution".to_string()),
        ),
        "local_code_tokens" => (
            "architecture_only".to_string(),
            false,
            "local_structure_search_only".to_string(),
            Some("local_code_tokens_not_real_eval_distribution".to_string()),
        ),
        "shard_root" => (
            "target_eval".to_string(),
            true,
            "promotion_and_submission_candidate".to_string(),
            None,
        ),
        _ => (
            "unknown".to_string(),
            false,
            "manual_inspection_required".to_string(),
            Some("unrecognized_data_root_layout".to_string()),
        ),
    };
    DataRootReport {
        requested_path,
        is_symlink,
        symlink_target,
        exists,
        resolved_path,
        kind,
        claim_tier,
        promotion_ready,
        recommended_use,
        blocked_reason,
        train_shard_count,
        val_shard_count,
    }
}

pub fn resolve_data_root(spec: Option<&str>) -> Result<ResolvedDataRoot, String> {
    let home = data_home_report();
    let input = spec.unwrap_or("@default").to_string();
    let key = input.trim();
    let alias_key = key.strip_prefix('@').unwrap_or(key);
    let effective_alias = if alias_key == "default" {
        Some(home.default_alias.clone())
    } else {
        home.aliases
            .iter()
            .find(|row| row.alias == alias_key)
            .map(|row| row.alias.clone())
    };
    let (resolution_kind, alias, selected_path) = if let Some(alias) = effective_alias {
        let row = home
            .aliases
            .iter()
            .find(|entry| entry.alias == alias)
            .ok_or_else(|| format!("data alias {alias} vanished during resolution"))?;
        ("alias".to_string(), Some(alias), row.path.clone())
    } else {
        ("direct_path".to_string(), None, input.clone())
    };
    let report = inspect_raw_data_root(Path::new(&selected_path));
    Ok(ResolvedDataRoot {
        input,
        resolution_kind,
        alias,
        selected_path,
        data_home: home.data_home,
        registry_path: home.registry_path,
        default_alias: home.default_alias,
        report,
    })
}

pub fn inspect_data_root(root: &Path) -> DataRootReport {
    inspect_raw_data_root(root)
}

pub fn materialize_data_root(root: &Path) -> Result<PathBuf, String> {
    let spec = root.display().to_string();
    let resolved = resolve_data_root(Some(&spec))?;
    Ok(PathBuf::from(resolved.selected_path))
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
    let root = materialize_data_root(root)?;
    let mut rows: Vec<PathBuf> = fs::read_dir(&root)
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
