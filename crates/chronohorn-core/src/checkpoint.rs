use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use chronohorn_runtime::{load_export_bundle_material, resolve_export_reference_path};
use zip::ZipArchive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Learned,
    StructuralControl,
    PackedMemory,
    DeterministicSubstrate,
    Unknown,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Learned => write!(f, "learned"),
            Self::StructuralControl => write!(f, "structural-control"),
            Self::PackedMemory => write!(f, "packed-memory"),
            Self::DeterministicSubstrate => write!(f, "deterministic-substrate"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_bytes: usize,
    pub category: Category,
}

pub fn inspect_npz(path: &str) -> Result<Vec<TensorEntry>, String> {
    let file = File::open(path).map_err(|err| format!("open {path}: {err}"))?;
    let mut archive = ZipArchive::new(file).map_err(|err| format!("zip open {path}: {err}"))?;
    let mut rows = Vec::new();
    for idx in 0..archive.len() {
        let mut entry = archive
            .by_index(idx)
            .map_err(|err| format!("zip entry {idx}: {err}"))?;
        if !entry.name().ends_with(".npy") {
            continue;
        }
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|err| format!("read {}: {err}", entry.name()))?;
        let meta = parse_npy_meta(&bytes)?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        rows.push(TensorEntry {
            category: classify_name(&name),
            name,
            dtype: meta.dtype,
            shape: meta.shape,
            data_bytes: meta.data_bytes,
        });
    }
    Ok(rows)
}

pub fn render_entries(entries: &[TensorEntry]) -> String {
    let mut out = String::new();
    out.push_str(&format!("tensor_count: {}\n", entries.len()));
    out.push_str("tensors:\n");
    for entry in entries {
        out.push_str(&format!(
            "  {:<48} {:<24} {:>10} bytes  {:<24} {}\n",
            entry.name,
            entry.category,
            entry.data_bytes,
            ShapeDisplay(&entry.shape),
            entry.dtype
        ));
    }
    out
}

#[derive(Debug, Clone)]
pub struct F32Array {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub fn load_named_f32_arrays(
    path: &str,
    names: &[&str],
) -> Result<BTreeMap<String, F32Array>, String> {
    let source = Path::new(path);
    if source.is_dir() {
        if source.join("manifest.json").is_file() {
            return load_named_export_bundle_f32_arrays(source, names);
        }
        return load_named_bundle_f32_arrays(source, names);
    }
    let extension = source.extension().and_then(|ext| ext.to_str());
    if extension == Some("npz") {
        return load_named_npz_f32_arrays(path, names);
    }
    Err(format!(
        "unsupported checkpoint source {path}: expected an .npz file or bundle directory"
    ))
}

fn load_named_export_bundle_f32_arrays(
    export_root: &Path,
    names: &[&str],
) -> Result<BTreeMap<String, F32Array>, String> {
    let loaded_bundle = load_export_bundle_material(export_root)?;
    let learned_state_index = loaded_bundle.learned_state_index;

    let by_name = learned_state_index
        .tensor_index
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect::<BTreeMap<_, _>>();

    let mut found = BTreeMap::new();
    for &name in names {
        let entry = by_name.get(name).ok_or_else(|| {
            format!(
                "missing tensor {name} in export bundle {}",
                export_root.display()
            )
        })?;
        let blob_path = resolve_export_reference_path(export_root, &entry.blob)?;
        let bytes = std::fs::read(&blob_path)
            .map_err(|err| format!("read {}: {err}", blob_path.display()))?;
        found.insert(
            name.to_string(),
            load_f32_array_from_npy_bytes(name, &bytes)?,
        );
    }
    Ok(found)
}

pub fn load_named_npz_f32_arrays(
    path: &str,
    names: &[&str],
) -> Result<BTreeMap<String, F32Array>, String> {
    let file = File::open(path).map_err(|err| format!("open {path}: {err}"))?;
    let mut archive = ZipArchive::new(file).map_err(|err| format!("zip open {path}: {err}"))?;
    let wanted: BTreeMap<&str, ()> = names.iter().copied().map(|name| (name, ())).collect();
    let mut found = BTreeMap::new();
    for idx in 0..archive.len() {
        let mut entry = archive
            .by_index(idx)
            .map_err(|err| format!("zip entry {idx}: {err}"))?;
        if !entry.name().ends_with(".npy") {
            continue;
        }
        let name = entry.name().trim_end_matches(".npy").to_string();
        if !wanted.contains_key(name.as_str()) {
            continue;
        }
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|err| format!("read {}: {err}", entry.name()))?;
        found.insert(name.clone(), load_f32_array_from_npy_bytes(&name, &bytes)?);
    }
    for &name in names {
        if !found.contains_key(name) {
            return Err(format!("missing tensor {name} in {path}"));
        }
    }
    Ok(found)
}

fn load_named_bundle_f32_arrays(
    bundle_dir: &Path,
    names: &[&str],
) -> Result<BTreeMap<String, F32Array>, String> {
    let mut found = BTreeMap::new();
    for &name in names {
        let tensor_path = resolve_bundle_tensor_path(bundle_dir, name)?;
        let bytes = std::fs::read(&tensor_path)
            .map_err(|err| format!("read {}: {err}", tensor_path.display()))?;
        found.insert(
            name.to_string(),
            load_f32_array_from_npy_bytes(name, &bytes)?,
        );
    }
    Ok(found)
}

fn resolve_bundle_tensor_path(bundle_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let candidate_file = format!("{name}.npy");
    let search_roots = bundle_tensor_search_roots(bundle_dir);
    for root in &search_roots {
        let candidate = root.join(&candidate_file);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let searched = search_roots
        .into_iter()
        .map(|root| root.join(&candidate_file).display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "missing tensor {name} in bundle {} (looked for {searched})",
        bundle_dir.display()
    ))
}

fn bundle_tensor_search_roots(bundle_dir: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    push_unique_path(&mut roots, bundle_dir.to_path_buf());
    if bundle_dir.file_name().and_then(|name| name.to_str()) == Some("artifacts") {
        if let Some(parent) = bundle_dir.parent() {
            push_unique_path(&mut roots, parent.to_path_buf());
        }
    } else {
        push_unique_path(&mut roots, bundle_dir.join("artifacts"));
    }
    roots
}

fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|path| path == &candidate) {
        paths.push(candidate);
    }
}

fn load_f32_array_from_npy_bytes(name: &str, bytes: &[u8]) -> Result<F32Array, String> {
    let meta = parse_npy_meta(bytes)?;
    if meta.dtype != "<f4" && meta.dtype != "|f4" {
        return Err(format!("unsupported dtype for {name}: {}", meta.dtype));
    }
    let elem_count = shape_elem_count(&meta.shape)?;
    let payload = &bytes[meta.data_offset..];
    let expected_bytes = elem_count
        .checked_mul(4)
        .ok_or_else(|| format!("byte overflow for {name}"))?;
    if payload.len() < expected_bytes {
        return Err(format!(
            "truncated payload for {name}: expected {expected_bytes} bytes, got {}",
            payload.len()
        ));
    }
    let mut values = Vec::with_capacity(elem_count);
    for chunk in payload[..expected_bytes].chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(F32Array {
        shape: meta.shape,
        values,
    })
}

fn classify_name(name: &str) -> Category {
    if name.contains("linear_kernel")
        || name.contains("linear_in_proj")
        || name.contains("linear_decays")
    {
        Category::DeterministicSubstrate
    } else if name.starts_with("packed_") {
        Category::PackedMemory
    } else if name.contains("causal_mask")
        || name.contains("recency_kernel")
        || name.ends_with("_mask")
        || name.contains("token_class_ids")
        || name.contains("vocab_axis")
        || name.contains("class_axis")
    {
        Category::StructuralControl
    } else if name.ends_with(".weight")
        || name.ends_with(".bias")
        || name.contains("readout")
        || name.contains("gate")
        || name.starts_with("student.")
        || name.starts_with("base.")
    {
        Category::Learned
    } else {
        Category::Unknown
    }
}

struct NpyMeta {
    dtype: String,
    shape: Vec<usize>,
    data_bytes: usize,
    data_offset: usize,
}

fn parse_npy_meta(bytes: &[u8]) -> Result<NpyMeta, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid npy magic".to_string());
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, data_offset) = match (major, minor) {
        (1, _) => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        (2, _) | (3, _) => {
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => return Err(format!("unsupported npy version {major}.{minor}")),
    };
    let header_end = data_offset + header_len;
    if header_end > bytes.len() {
        return Err("truncated npy header".to_string());
    }
    let header = std::str::from_utf8(&bytes[data_offset..header_end])
        .map_err(|err| format!("header utf8: {err}"))?;
    let dtype = extract_between(header, "'descr':", ",")
        .or_else(|| extract_between(header, "\"descr\":", ","))
        .unwrap_or_else(|| "?".to_string())
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .to_string();
    let shape_block =
        extract_paren_block(header, "shape").ok_or_else(|| "missing shape".to_string())?;
    let shape = parse_shape(&shape_block)?;
    let data_bytes = bytes.len().saturating_sub(header_end);
    Ok(NpyMeta {
        dtype,
        shape,
        data_bytes,
        data_offset: header_end,
    })
}

fn extract_between(text: &str, start_pat: &str, end_pat: &str) -> Option<String> {
    let start = text.find(start_pat)? + start_pat.len();
    let rest = &text[start..];
    let end = rest.find(end_pat)?;
    Some(rest[..end].to_string())
}

fn extract_paren_block(text: &str, key: &str) -> Option<String> {
    let key_pos = text.find(key)?;
    let open_rel = text[key_pos..].find('(')?;
    let open = key_pos + open_rel;
    let close_rel = text[open..].find(')')?;
    let close = open + close_rel;
    Some(text[open + 1..close].to_string())
}

fn parse_shape(raw: &str) -> Result<Vec<usize>, String> {
    let mut dims = Vec::new();
    for piece in raw.split(',') {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        dims.push(
            trimmed
                .parse::<usize>()
                .map_err(|err| format!("shape dim {trimmed}: {err}"))?,
        );
    }
    Ok(dims)
}

fn shape_elem_count(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut total = 1usize;
    for &dim in shape {
        total = total
            .checked_mul(dim)
            .ok_or_else(|| format!("shape overflow for {:?}", shape))?;
    }
    Ok(total)
}

struct ShapeDisplay<'a>(&'a [usize]);

impl fmt::Display for ShapeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (idx, dim) in self.0.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, ")")
    }
}
