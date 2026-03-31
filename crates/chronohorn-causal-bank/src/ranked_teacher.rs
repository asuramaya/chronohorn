use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const BLINX_RANKED_ORACLE_SCHEMA_VERSION: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct RankedTeacherCandidates {
    pub tokens: Vec<usize>,
    pub counts: Vec<usize>,
    pub frequencies: Vec<f64>,
}

impl RankedTeacherCandidates {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn total_count(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn pairs(&self) -> Vec<(usize, usize)> {
        self.tokens
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.tokens
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .zip(self.frequencies.iter().copied())
            .map(|((token, count), frequency)| (token, count, frequency))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedTeacherProvenance {
    pub path: String,
    pub size: usize,
    pub radius: usize,
    pub position: usize,
    pub center: u8,
    pub center_hex: String,
    pub left_context_hex: String,
    pub right_context_hex: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedTeacherRecord {
    pub schema_version: usize,
    pub provenance: RankedTeacherProvenance,
    pub bidi_inclusive_support_size: usize,
    pub bidi_leaveout_support_size: usize,
    pub left_leaveout_support_size: usize,
    pub candidates: RankedTeacherCandidates,
    pub bidi_inclusive_candidate4: bool,
    pub bidi_leaveout_candidate4: bool,
    pub left_leaveout_candidate4: bool,
    pub candidate_set_leq_4: bool,
    pub candidate_set_leq_8: bool,
    pub required_radius: Option<usize>,
    pub future_uplift: i64,
    pub self_inclusion_uplift: i64,
    pub clean_bridge_score: f64,
    pub memory_trust: f64,
    pub bridge_confidence: f64,
    pub self_inclusion_support_changed: bool,
    pub future_context_support_changed: bool,
}

impl RankedTeacherRecord {
    pub fn candidate_pairs(&self) -> Vec<(usize, usize)> {
        self.candidates.pairs()
    }

    pub fn teacher_support_size(&self) -> usize {
        self.candidates.len()
    }
}

pub struct RankedTeacherJsonlIter<R> {
    reader: R,
    source_name: String,
    line_buf: String,
    next_line_no: usize,
}

impl<R> RankedTeacherJsonlIter<R> {
    fn new(reader: R, source_name: String) -> Self {
        Self {
            reader,
            source_name,
            line_buf: String::new(),
            next_line_no: 1,
        }
    }
}

impl<R: BufRead> Iterator for RankedTeacherJsonlIter<R> {
    type Item = Result<RankedTeacherRecord, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => return None,
                Ok(_) => {
                    let line_no = self.next_line_no;
                    self.next_line_no += 1;
                    let line = self.line_buf.trim();
                    if line.is_empty() {
                        continue;
                    }
                    return Some(parse_ranked_teacher_line(
                        line,
                        &self.source_name,
                        Some(line_no),
                    ));
                }
                Err(err) => {
                    return Some(Err(format!(
                        "read ranked teacher {}:{}: {err}",
                        self.source_name, self.next_line_no
                    )));
                }
            }
        }
    }
}

pub fn iter_ranked_teacher_jsonl<R: BufRead>(
    reader: R,
    source_name: impl Into<String>,
) -> RankedTeacherJsonlIter<R> {
    RankedTeacherJsonlIter::new(reader, source_name.into())
}

pub fn open_ranked_teacher_jsonl<P: AsRef<Path>>(
    path: P,
) -> Result<RankedTeacherJsonlIter<BufReader<File>>, String> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref).map_err(|err| format!("open {}: {err}", path_ref.display()))?;
    Ok(iter_ranked_teacher_jsonl(
        BufReader::new(file),
        path_ref.display().to_string(),
    ))
}

pub fn load_ranked_teacher_jsonl<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<RankedTeacherRecord>, String> {
    let path_ref = path.as_ref();
    let raw = std::fs::read_to_string(path_ref)
        .map_err(|err| format!("read {}: {err}", path_ref.display()))?;
    parse_ranked_teacher_records(&raw, &path_ref.display().to_string())
}

pub fn parse_ranked_teacher_records(
    raw: &str,
    source_name: &str,
) -> Result<Vec<RankedTeacherRecord>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if let Ok(rows) = serde_json::from_str::<Vec<RawRankedTeacherRecord>>(trimmed) {
        return rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| validate_ranked_teacher_record(row, source_name, Some(index + 1)))
            .collect();
    }

    let mut rows = Vec::new();
    for (line_index, line) in trimmed.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rows.push(parse_ranked_teacher_line(
            line,
            source_name,
            Some(line_index + 1),
        )?);
    }
    if rows.is_empty() {
        return Err(format!("no ranked teacher rows found in {source_name}"));
    }
    Ok(rows)
}

fn parse_ranked_teacher_line(
    line: &str,
    source_name: &str,
    line_no: Option<usize>,
) -> Result<RankedTeacherRecord, String> {
    let row =
        serde_json::from_str::<RawRankedTeacherRecord>(line).map_err(|err| match line_no {
            Some(line_no) => format!("parse ranked teacher {source_name}:{line_no}: {err}"),
            None => format!("parse ranked teacher {source_name}: {err}"),
        })?;
    validate_ranked_teacher_record(row, source_name, line_no)
}

#[derive(Debug, Clone, Deserialize)]
struct RawRankedTeacherRecord {
    schema_version: usize,
    path: String,
    size: usize,
    radius: usize,
    position: usize,
    center: u8,
    center_hex: String,
    left_context_hex: String,
    right_context_hex: String,
    bidi_inclusive_support_size: usize,
    bidi_leaveout_support_size: usize,
    left_leaveout_support_size: usize,
    teacher_candidate_tokens: Vec<usize>,
    teacher_candidate_counts: Vec<usize>,
    teacher_candidate_frequencies: Vec<f64>,
    bidi_inclusive_candidate4: bool,
    bidi_leaveout_candidate4: bool,
    left_leaveout_candidate4: bool,
    candidate_set_leq_4: bool,
    candidate_set_leq_8: bool,
    required_radius: Option<usize>,
    future_uplift: i64,
    self_inclusion_uplift: i64,
    clean_bridge_score: f64,
    memory_trust: f64,
    bridge_confidence: f64,
    self_inclusion_support_changed: bool,
    future_context_support_changed: bool,
}

fn validate_ranked_teacher_record(
    raw: RawRankedTeacherRecord,
    source_name: &str,
    line_no: Option<usize>,
) -> Result<RankedTeacherRecord, String> {
    let context = match line_no {
        Some(line_no) => format!("{source_name}:{line_no}"),
        None => source_name.to_string(),
    };

    if raw.schema_version != BLINX_RANKED_ORACLE_SCHEMA_VERSION {
        return Err(format!(
            "ranked teacher {context} schema_version {} != {}",
            raw.schema_version, BLINX_RANKED_ORACLE_SCHEMA_VERSION
        ));
    }
    if raw.radius == 0 {
        return Err(format!("ranked teacher {context} radius must be positive"));
    }
    if raw.path.trim().is_empty() {
        return Err(format!("ranked teacher {context} path must not be empty"));
    }
    validate_hex_field("center_hex", &raw.center_hex, Some(1), &context)?;
    validate_hex_field(
        "left_context_hex",
        &raw.left_context_hex,
        Some(raw.radius),
        &context,
    )?;
    validate_hex_field(
        "right_context_hex",
        &raw.right_context_hex,
        Some(raw.radius),
        &context,
    )?;
    let parsed_center = u8::from_str_radix(&raw.center_hex, 16).map_err(|err| {
        format!(
            "ranked teacher {context} center_hex {:?} is not valid hex: {err}",
            raw.center_hex
        )
    })?;
    if parsed_center != raw.center {
        return Err(format!(
            "ranked teacher {context} center {} != center_hex {}",
            raw.center, raw.center_hex
        ));
    }
    if let Some(required_radius) = raw.required_radius {
        if required_radius == 0 {
            return Err(format!(
                "ranked teacher {context} required_radius must be positive when present"
            ));
        }
    }

    if raw.teacher_candidate_tokens.len() != raw.teacher_candidate_counts.len()
        || raw.teacher_candidate_tokens.len() != raw.teacher_candidate_frequencies.len()
    {
        return Err(format!(
            "ranked teacher {context} candidate length mismatch: tokens {} counts {} freqs {}",
            raw.teacher_candidate_tokens.len(),
            raw.teacher_candidate_counts.len(),
            raw.teacher_candidate_frequencies.len()
        ));
    }
    if raw.teacher_candidate_tokens.len() != raw.bidi_leaveout_support_size {
        return Err(format!(
            "ranked teacher {context} candidate length {} != bidi_leaveout_support_size {}",
            raw.teacher_candidate_tokens.len(),
            raw.bidi_leaveout_support_size
        ));
    }

    let total_count = raw.teacher_candidate_counts.iter().sum::<usize>();
    if raw.teacher_candidate_tokens.is_empty() {
        if total_count != 0 {
            return Err(format!(
                "ranked teacher {context} empty candidate support had non-zero total count {total_count}"
            ));
        }
    } else if total_count == 0 {
        return Err(format!(
            "ranked teacher {context} non-empty candidate support had zero total count"
        ));
    }

    let mut seen_tokens = HashSet::new();
    let mut prev_token = None;
    let mut prev_count = None;
    for ((&token, &count), &frequency) in raw
        .teacher_candidate_tokens
        .iter()
        .zip(raw.teacher_candidate_counts.iter())
        .zip(raw.teacher_candidate_frequencies.iter())
    {
        if !seen_tokens.insert(token) {
            return Err(format!(
                "ranked teacher {context} duplicated candidate token {token}"
            ));
        }
        if count == 0 {
            return Err(format!(
                "ranked teacher {context} candidate token {token} had zero count"
            ));
        }
        if !frequency.is_finite() || frequency < 0.0 {
            return Err(format!(
                "ranked teacher {context} candidate token {token} had invalid frequency {frequency}"
            ));
        }
        let expected_frequency = count as f64 / total_count as f64;
        if !approx_eq(frequency, expected_frequency, 1e-9) {
            return Err(format!(
                "ranked teacher {context} candidate token {token} frequency {frequency} != expected {expected_frequency}"
            ));
        }
        if let (Some(prev_token), Some(prev_count)) = (prev_token, prev_count) {
            if prev_count < count || (prev_count == count && prev_token > token) {
                return Err(format!(
                    "ranked teacher {context} candidates were not ranked by (-count, token)"
                ));
            }
        }
        prev_token = Some(token);
        prev_count = Some(count);
    }

    let expected_bidi_inclusive_candidate4 = raw.bidi_inclusive_support_size <= 4;
    if raw.bidi_inclusive_candidate4 != expected_bidi_inclusive_candidate4 {
        return Err(format!(
            "ranked teacher {context} bidi_inclusive_candidate4 {} != expected {}",
            raw.bidi_inclusive_candidate4, expected_bidi_inclusive_candidate4
        ));
    }
    let expected_bidi_leaveout_candidate4 =
        raw.bidi_leaveout_support_size > 0 && raw.bidi_leaveout_support_size <= 4;
    if raw.bidi_leaveout_candidate4 != expected_bidi_leaveout_candidate4 {
        return Err(format!(
            "ranked teacher {context} bidi_leaveout_candidate4 {} != expected {}",
            raw.bidi_leaveout_candidate4, expected_bidi_leaveout_candidate4
        ));
    }
    let expected_left_leaveout_candidate4 =
        raw.left_leaveout_support_size > 0 && raw.left_leaveout_support_size <= 4;
    if raw.left_leaveout_candidate4 != expected_left_leaveout_candidate4 {
        return Err(format!(
            "ranked teacher {context} left_leaveout_candidate4 {} != expected {}",
            raw.left_leaveout_candidate4, expected_left_leaveout_candidate4
        ));
    }
    let expected_candidate_set_leq_4 = raw.bidi_inclusive_support_size <= 4;
    if raw.candidate_set_leq_4 != expected_candidate_set_leq_4 {
        return Err(format!(
            "ranked teacher {context} candidate_set_leq_4 {} != expected {}",
            raw.candidate_set_leq_4, expected_candidate_set_leq_4
        ));
    }
    let expected_candidate_set_leq_8 = raw.bidi_inclusive_support_size <= 8;
    if raw.candidate_set_leq_8 != expected_candidate_set_leq_8 {
        return Err(format!(
            "ranked teacher {context} candidate_set_leq_8 {} != expected {}",
            raw.candidate_set_leq_8, expected_candidate_set_leq_8
        ));
    }

    let expected_future_uplift = i64::from(raw.bidi_leaveout_support_size == 1)
        - i64::from(raw.left_leaveout_support_size == 1);
    if raw.future_uplift != expected_future_uplift {
        return Err(format!(
            "ranked teacher {context} future_uplift {} != expected {}",
            raw.future_uplift, expected_future_uplift
        ));
    }
    let expected_self_inclusion_uplift = i64::from(raw.bidi_inclusive_support_size == 1)
        - i64::from(raw.bidi_leaveout_support_size == 1);
    if raw.self_inclusion_uplift != expected_self_inclusion_uplift {
        return Err(format!(
            "ranked teacher {context} self_inclusion_uplift {} != expected {}",
            raw.self_inclusion_uplift, expected_self_inclusion_uplift
        ));
    }

    let expected_memory_trust = support_trust(raw.left_leaveout_support_size);
    if !approx_eq(raw.memory_trust, expected_memory_trust, 1e-12) {
        return Err(format!(
            "ranked teacher {context} memory_trust {} != expected {}",
            raw.memory_trust, expected_memory_trust
        ));
    }
    let expected_bridge_confidence = support_trust(raw.bidi_leaveout_support_size);
    if !approx_eq(raw.bridge_confidence, expected_bridge_confidence, 1e-12) {
        return Err(format!(
            "ranked teacher {context} bridge_confidence {} != expected {}",
            raw.bridge_confidence, expected_bridge_confidence
        ));
    }
    let expected_clean_bridge_score =
        harmonic_mean(expected_memory_trust, expected_bridge_confidence);
    if !approx_eq(raw.clean_bridge_score, expected_clean_bridge_score, 1e-12) {
        return Err(format!(
            "ranked teacher {context} clean_bridge_score {} != expected {}",
            raw.clean_bridge_score, expected_clean_bridge_score
        ));
    }

    let expected_self_inclusion_support_changed =
        raw.bidi_inclusive_support_size != raw.bidi_leaveout_support_size;
    if raw.self_inclusion_support_changed != expected_self_inclusion_support_changed {
        return Err(format!(
            "ranked teacher {context} self_inclusion_support_changed {} != expected {}",
            raw.self_inclusion_support_changed, expected_self_inclusion_support_changed
        ));
    }
    let expected_future_context_support_changed =
        raw.bidi_leaveout_support_size != raw.left_leaveout_support_size;
    if raw.future_context_support_changed != expected_future_context_support_changed {
        return Err(format!(
            "ranked teacher {context} future_context_support_changed {} != expected {}",
            raw.future_context_support_changed, expected_future_context_support_changed
        ));
    }

    Ok(RankedTeacherRecord {
        schema_version: raw.schema_version,
        provenance: RankedTeacherProvenance {
            path: raw.path,
            size: raw.size,
            radius: raw.radius,
            position: raw.position,
            center: raw.center,
            center_hex: raw.center_hex,
            left_context_hex: raw.left_context_hex,
            right_context_hex: raw.right_context_hex,
        },
        bidi_inclusive_support_size: raw.bidi_inclusive_support_size,
        bidi_leaveout_support_size: raw.bidi_leaveout_support_size,
        left_leaveout_support_size: raw.left_leaveout_support_size,
        candidates: RankedTeacherCandidates {
            tokens: raw.teacher_candidate_tokens,
            counts: raw.teacher_candidate_counts,
            frequencies: raw.teacher_candidate_frequencies,
        },
        bidi_inclusive_candidate4: raw.bidi_inclusive_candidate4,
        bidi_leaveout_candidate4: raw.bidi_leaveout_candidate4,
        left_leaveout_candidate4: raw.left_leaveout_candidate4,
        candidate_set_leq_4: raw.candidate_set_leq_4,
        candidate_set_leq_8: raw.candidate_set_leq_8,
        required_radius: raw.required_radius,
        future_uplift: raw.future_uplift,
        self_inclusion_uplift: raw.self_inclusion_uplift,
        clean_bridge_score: raw.clean_bridge_score,
        memory_trust: raw.memory_trust,
        bridge_confidence: raw.bridge_confidence,
        self_inclusion_support_changed: raw.self_inclusion_support_changed,
        future_context_support_changed: raw.future_context_support_changed,
    })
}

fn validate_hex_field(
    field_name: &str,
    value: &str,
    expected_bytes: Option<usize>,
    context: &str,
) -> Result<(), String> {
    if value.len() % 2 != 0 {
        return Err(format!(
            "ranked teacher {context} {field_name} {:?} had odd hex length",
            value
        ));
    }
    if let Some(expected_bytes) = expected_bytes {
        let expected_len = expected_bytes * 2;
        if value.len() != expected_len {
            return Err(format!(
                "ranked teacher {context} {field_name} {:?} had length {}, expected {}",
                value,
                value.len(),
                expected_len
            ));
        }
    }
    if !value.as_bytes().iter().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "ranked teacher {context} {field_name} {:?} was not valid hex",
            value
        ));
    }
    Ok(())
}

fn support_trust(support_size: usize) -> f64 {
    if support_size > 0 {
        1.0 / support_size as f64
    } else {
        0.0
    }
}

fn harmonic_mean(left: f64, right: f64) -> f64 {
    if left > 0.0 && right > 0.0 {
        (2.0 * left * right) / (left + right)
    } else {
        0.0
    }
}

fn approx_eq(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance.max(f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_record_json() -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":3,",
                "\"path\":\"sample.bin\",",
                "\"size\":128,",
                "\"radius\":2,",
                "\"position\":17,",
                "\"center\":42,",
                "\"center_hex\":\"2a\",",
                "\"left_context_hex\":\"abcd\",",
                "\"right_context_hex\":\"ef01\",",
                "\"bidi_inclusive_support_size\":3,",
                "\"bidi_leaveout_support_size\":2,",
                "\"left_leaveout_support_size\":1,",
                "\"teacher_candidate_tokens\":[41,9],",
                "\"teacher_candidate_counts\":[5,2],",
                "\"teacher_candidate_frequencies\":[{freq0},{freq1}],",
                "\"bidi_inclusive_candidate4\":true,",
                "\"bidi_leaveout_candidate4\":true,",
                "\"left_leaveout_candidate4\":true,",
                "\"candidate_set_leq_4\":true,",
                "\"candidate_set_leq_8\":true,",
                "\"required_radius\":1,",
                "\"future_uplift\":-1,",
                "\"self_inclusion_uplift\":0,",
                "\"clean_bridge_score\":{clean_bridge_score},",
                "\"memory_trust\":1.0,",
                "\"bridge_confidence\":0.5,",
                "\"self_inclusion_support_changed\":true,",
                "\"future_context_support_changed\":true",
                "}}"
            ),
            freq0 = 5.0 / 7.0,
            freq1 = 2.0 / 7.0,
            clean_bridge_score = 2.0 / 3.0,
        )
    }

    #[test]
    fn parses_single_jsonl_record() {
        let rows = parse_ranked_teacher_records(&sample_record_json(), "inline").unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.provenance.path, "sample.bin");
        assert_eq!(row.provenance.center, 42);
        assert_eq!(row.candidate_pairs(), vec![(41, 5), (9, 2)]);
        assert_eq!(row.teacher_support_size(), 2);
        assert!((row.candidates.frequencies[0] - 5.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn parses_json_array_for_loader_convenience() {
        let raw = format!("[{}]", sample_record_json());
        let rows = parse_ranked_teacher_records(&raw, "inline").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].candidate_pairs(), vec![(41, 5), (9, 2)]);
    }

    #[test]
    fn iterator_skips_blank_lines() {
        let raw = format!("\n{}\n\n", sample_record_json());
        let rows = iter_ranked_teacher_jsonl(Cursor::new(raw), "inline")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].provenance.position, 17);
    }

    #[test]
    fn rejects_candidate_length_mismatch() {
        let raw = sample_record_json().replace(
            "\"teacher_candidate_frequencies\":[0.7142857142857143,0.2857142857142857]",
            "\"teacher_candidate_frequencies\":[1.0]",
        );
        let err = parse_ranked_teacher_records(&raw, "inline").unwrap_err();
        assert!(err.contains("candidate length mismatch"));
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let raw = sample_record_json().replace("\"schema_version\":3", "\"schema_version\":2");
        let err = parse_ranked_teacher_records(&raw, "inline").unwrap_err();
        assert!(err.contains("schema_version 2 !="));
    }
}
