use std::fmt;
use std::fs::File;
use std::io::Read;

use zip::ZipArchive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Learned,
    StructuralControl,
    DeterministicSubstrate,
    Unknown,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Learned => write!(f, "learned"),
            Self::StructuralControl => write!(f, "structural-control"),
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
        let mut entry = archive.by_index(idx).map_err(|err| format!("zip entry {idx}: {err}"))?;
        if !entry.name().ends_with(".npy") {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)
            .map_err(|err| format!("read {}: {err}", entry.name()))?;
        let meta = parse_npy_header(&bytes)?;
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

fn classify_name(name: &str) -> Category {
    if name.contains("linear_kernel")
        || name.contains("linear_in_proj")
        || name.contains("linear_decays")
    {
        Category::DeterministicSubstrate
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
}

fn parse_npy_header(bytes: &[u8]) -> Result<NpyMeta, String> {
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
    let shape_block = extract_paren_block(header, "shape").ok_or_else(|| "missing shape".to_string())?;
    let shape = parse_shape(&shape_block)?;
    let data_bytes = bytes.len().saturating_sub(header_end);
    Ok(NpyMeta {
        dtype,
        shape,
        data_bytes,
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

