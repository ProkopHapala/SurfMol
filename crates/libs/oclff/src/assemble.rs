//! OpenCL source assembler — Rust port of SPAMMM `OpenCLBase`.
//!
//! Parses `.cl` "libraries" for `//>>>function` / `//>>>macro` blocks and
//! preprocesses template `.cl` files containing `//<<<` sentinels:
//!   - `//<<<file FRAGMENT_NAME`  — inject entire fragment (exact line)
//!   - `//<<<macro MARKER`        — inject macro body (exact line)
//!   - `//<<<function MARKER(...)` — replace marker with a function name
//!     (every occurrence of the exact string `//<<<function MARKER` is replaced)
//!
//! Fragments are supplied by name → source string. In the Rust harness the
//! sources are typically loaded at compile time via `include_str!` so the
//! assembler is runtime-only (parsing + string replacement).

use std::collections::HashMap;
use std::path::Path;

/// A parsed `.cl` library: named `//>>>function` and `//>>>macro` blocks.
#[derive(Debug, Default, Clone)]
pub struct ClLibrary {
    pub functions: HashMap<String, String>,
    pub macros: HashMap<String, String>,
}

/// Substitution table for a template.
#[derive(Debug, Default, Clone)]
pub struct Substitutions {
    /// `//<<<function MARKER` → replacement string (typically a function name).
    pub functions: HashMap<String, String>,
    /// `//<<<macro MARKER` → replacement string (macro body / arbitrary text).
    pub macros: HashMap<String, String>,
    /// `//<<<file NAME` → already resolved; normally use `Assembler::add_fragment` instead.
    pub files: HashMap<String, String>,
}

impl Substitutions {
    pub fn new() -> Self { Self::default() }
}

impl ClLibrary {
    /// Parse a `.cl` library into `//>>>function` / `//>>>macro` blocks.
    /// Blocks end at the next `//>>>` or end-of-source.
    pub fn parse(src: &str) -> Self {
        let mut lib = ClLibrary::default();
        let mut kind: Option<&str> = None;
        let mut name: Option<String> = None;
        let mut body: Vec<&str> = Vec::new();

        for line in src.lines() {
            let stripped = line.trim();
            if stripped.starts_with("//>>>") {
                if let (Some(k), Some(n), b) = (kind, name.take(), body.drain(..).collect::<Vec<_>>()) {
                    let text = b.join("\n");
                    if k == "function" { lib.functions.insert(n, text); }
                    else { lib.macros.insert(n, text); }
                }
                let header = &stripped[5..].trim();
                if let Some(rest) = header.strip_prefix("function") {
                    let name_part = rest.trim();
                    kind = Some("function");
                    let key = name_part.split('(').next().unwrap_or(name_part).trim().to_string();
                    name = Some(key);
                } else if let Some(rest) = header.strip_prefix("macro") {
                    let name_part = rest.trim();
                    kind = Some("macro");
                    let key = name_part.split_whitespace().next().unwrap_or(name_part).to_string();
                    name = Some(key);
                } else {
                    kind = Some("macro");
                    name = Some(header.to_string());
                }
            } else if kind.is_some() {
                body.push(line);
            }
        }

        if let (Some(k), Some(n), b) = (kind, name, body.drain(..).collect::<Vec<_>>()) {
            let text = b.join("\n");
            if k == "function" { lib.functions.insert(n, text); }
            else { lib.macros.insert(n, text); }
        }

        lib
    }

    /// Build substitutions from all macros in this library.
    pub fn macro_subs(&self) -> HashMap<String, String> { self.macros.clone() }
    pub fn function_subs(&self) -> HashMap<String, String> { self.functions.clone() }
}

/// Stores named OpenCL fragment sources and their parsed `//>>>` libraries,
/// and preprocesses templates with `//<<<` sentinels.
#[derive(Debug, Default, Clone)]
pub struct ClAssembler {
    fragments: HashMap<String, String>,
    libs: HashMap<String, ClLibrary>,
}

impl ClAssembler {
    pub fn new() -> Self { Self::default() }

    /// Add a named fragment source (e.g. `("common.cl", include_str!(...))`).
    /// The source is parsed for `//>>>` blocks and stored under `name`.
    pub fn add_fragment(&mut self, name: impl Into<String>, src: impl Into<String>) -> &ClLibrary {
        let name = name.into();
        let src = src.into();
        let lib = ClLibrary::parse(&src);
        self.libs.insert(name.clone(), lib);
        self.fragments.insert(name.clone(), src);
        self.libs.get(&name).unwrap()
    }

    /// Load a fragment from a file path and add it under the file stem name.
    pub fn add_fragment_from_path(&mut self, path: impl AsRef<Path>) -> Result<&ClLibrary, std::io::Error> {
        let path = path.as_ref();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let src = std::fs::read_to_string(path)?;
        Ok(self.add_fragment(name, src))
    }

    pub fn fragment(&self, name: &str) -> Option<&str> { self.fragments.get(name).map(|s| s.as_str()) }
    pub fn library(&self, name: &str) -> Option<&ClLibrary> { self.libs.get(name) }

    /// Preprocess a template string.
    ///
    /// - `//<<<file FRAGMENT_NAME` (exact stripped line) is replaced by the
    ///   registered fragment of that name.
    /// - `//<<<macro MARKER` (exact stripped line) is replaced by `subs.macros[MARKER]`.
    /// - `//<<<function MARKER` is a prefix replacement: every occurrence of
    ///   the exact string `//<<<function MARKER` is replaced by `subs.functions[MARKER]`.
    pub fn assemble(&self, template: &str, subs: &Substitutions) -> Result<String, String> {
        let mut out_lines: Vec<String> = Vec::new();

        for line in template.lines() {
            let stripped = line.trim();
            if let Some(fname) = stripped.strip_prefix("//<<<file ") {
                let fname = fname.trim();
                match self.fragments.get(fname) {
                    Some(content) => out_lines.push(content.clone()),
                    None => return Err(format!("ClAssembler: missing fragment for //<<<file {fname}")),
                }
            } else if let Some(mname) = stripped.strip_prefix("//<<<macro ") {
                let mname = mname.trim();
                match subs.macros.get(mname) {
                    Some(body) => {
                        for bline in body.lines() { out_lines.push(bline.to_string()); }
                    }
                    None => return Err(format!("ClAssembler: missing macro substitution for //<<<macro {mname}")),
                }
            } else {
                out_lines.push(line.to_string());
            }
        }

        let mut out = out_lines.join("\n");

        // Function-name substitutions: exact string replacement across the whole source.
        for (marker, replacement) in &subs.functions {
            let needle = format!("//<<<function {marker}");
            out = out.replace(&needle, replacement);
        }

        // File substitutions from subs.files (overrides fragment map, same mechanism).
        for (marker, content) in &subs.files {
            let needle = format!("//<<<file {marker}");
            out = out.replace(&needle, content);
        }

        Ok(out)
    }

    /// Assemble a template into a complete OpenCL source string.
    ///
    /// The returned source can then be compiled with `ocl::ProQue::builder().src(out).build()`.
    pub fn assemble_program(&self, template: &str, subs: &Substitutions) -> Result<String, String> {
        self.assemble(template, subs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORCES_LIB: &str = r#"
//>>>function getLJQH (dp,REQ,ffpars)
inline float4 getLJQH(float3 dp, float4 REQ, float R2damp) { return (float4)(0.0f); }

//>>>macro MODEL_MorseQ_PAIR
{
    float e = exp(K*(r-R0));
    float4 fe = (float4)(0.0f);
}
"#;

    #[test]
    fn parse_lib() {
        let lib = ClLibrary::parse(FORCES_LIB);
        assert!(lib.functions.contains_key("getLJQH"));
        assert!(lib.macros.contains_key("MODEL_MorseQ_PAIR"));
        assert!(lib.functions["getLJQH"].contains("inline float4 getLJQH"));
    }

    #[test]
    fn assemble_file_and_macro() {
        let mut asm = ClAssembler::new();
        asm.add_fragment("common.cl", "#define FOO 1\n");
        asm.add_fragment("Forces.cl", FORCES_LIB);

        let template = r#"//<<<file common.cl
//<<<file Forces.cl
__kernel void foo(){
    float r = 1.0f;
    //<<<macro MODEL_MorseQ_PAIR
}
"#;

        let mut subs = Substitutions::new();
        subs.macros.insert("MODEL_MorseQ_PAIR".to_string(), asm.library("Forces.cl").unwrap().macros["MODEL_MorseQ_PAIR"].clone());

        let out = asm.assemble(template, &subs).unwrap();
        assert!(out.contains("#define FOO 1"));
        assert!(out.contains("inline float4 getLJQH"));
        assert!(out.contains("float e = exp(K*(r-R0))"));
    }

    #[test]
    fn assemble_function_name() {
        let mut asm = ClAssembler::new();
        asm.add_fragment("Forces.cl", FORCES_LIB);

        let template = "__kernel void foo(){ float4 fe = //<<<function getLJQH(dp, REQ, 1.0f); }";
        let mut subs = Substitutions::new();
        subs.functions.insert("getLJQH".to_string(), "getMorseQH".to_string());

        let out = asm.assemble(template, &subs).unwrap();
        assert!(out.contains("float4 fe = getMorseQH(dp, REQ, 1.0f);"));
    }
}
