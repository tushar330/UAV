const fs = require('fs');
const path = require('path');
const D = require('docx');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
  TableOfContents, LevelFormat, PageOrientation, convertInchesToTwip,
} = D;

// Repo-relative: this file lives at <repo>/tools/viva_docx/
const ROOT = path.resolve(__dirname, '..', '..');
const SRC = path.join(ROOT, 'VIVA_STUDY_GUIDE.md');
const OUT = path.join(ROOT, 'BTP_Viva_Study_Guide.docx');
const FIGDIR = path.join(__dirname, '.figcache');

if (!fs.existsSync(path.join(FIGDIR, 'manifest.json'))) {
  console.error('Missing figure cache. Run first:\n  python tools/viva_docx/prepare_figures.py');
  process.exit(1);
}
const FIGS = JSON.parse(fs.readFileSync(path.join(FIGDIR, 'manifest.json'), 'utf8'));

// US Letter, 0.75in margins
const PAGE_W = 12240, MARGIN = 1080;
const CONTENT = PAGE_W - 2 * MARGIN;      // 10080 DXA
const IMG_W = 660;                         // px @96dpi ~ 6.9in

const ACCENT = '0F6E5C', ALERT = 'B4451F', INK = '161A19', MUTED = '5E6B66';
const SHADE = 'EFF2EE', SHADE2 = 'E2EFEA';

/* ---------------- inline math -> readable unicode ---------------- */
const GREEK = {alpha:'α',beta:'β',gamma:'γ',delta:'δ',Delta:'Δ',epsilon:'ε',eta:'η',
  theta:'θ',lambda:'λ',mu:'μ',rho:'ρ',sigma:'σ',tau:'τ',phi:'φ',omega:'ω',Omega:'Ω',pi:'π'};
const SUP = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹',
  '+':'⁺','-':'⁻','n':'ⁿ','i':'ⁱ','a':'ᵃ','α':'ᵅ'};
const SUB = {'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉',
  'i':'ᵢ','c':'꜀','a':'ₐ','k':'ₖ','S':'ₛ'};

// balanced-brace group starting at s[i] === '{'
function takeGroup(s, i) {
  let d = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === '{') d++;
    else if (s[j] === '}') { d--; if (!d) return [s.slice(i + 1, j), j + 1]; }
  }
  return [s.slice(i + 1), s.length];
}
// expand \name{..}{..} with balanced braces (regex cannot nest)
function expandCmd(t, name, n, fmt) {
  let out = '', i = 0;
  for (;;) {
    const k = t.indexOf('\\' + name, i);
    if (k < 0) { out += t.slice(i); return out; }
    out += t.slice(i, k);
    let j = k + name.length + 1;
    const args = [];
    for (let a = 0; a < n; a++) {
      while (j < t.length && t[j] === ' ') j++;
      if (t[j] !== '{') break;
      const [g, nj] = takeGroup(t, j); args.push(g); j = nj;
    }
    if (args.length < n) { out += t.slice(k, j); i = j; continue; }
    out += fmt(...args);
    i = j;
  }
}

// \underbrace{X}_{label}  ->  "X [label]"  (labels carry meaning, so keep them)
function expandUnderbrace(t) {
  const TAG = '\\underbrace';
  let out = '', i = 0;
  for (;;) {
    const k = t.indexOf(TAG, i);
    if (k < 0) { out += t.slice(i); return out; }
    out += t.slice(i, k);
    let j = k + TAG.length;
    while (j < t.length && t[j] === ' ') j++;
    if (t[j] !== '{') { out += t.slice(k, j); i = j; continue; }
    const [content, j2] = takeGroup(t, j);
    j = j2;
    let label = '';
    if (t[j] === '_') {
      let m = j + 1;
      while (m < t.length && t[m] === ' ') m++;
      if (t[m] === '{') { const [lb, j3] = takeGroup(t, m); label = lb; j = j3; }
    }
    out += content + (label ? ' [' + label + ']' : '');
    i = j;
  }
}

function mathToText(s) {
  let t = s;
  t = t.replace(/\\(?:left|right|bigg|Bigg|big|Big|quad|qquad)\b/g, ' ');
  t = t.replace(/\\[,;!:]/g, ' ');
  t = t.replace(/\\[dt]?frac(\d)(\d)/g, '($1)/($2)');
  t = expandUnderbrace(t);
  for (let p = 0; p < 5; p++) {
    const before = t;
    t = expandCmd(t, 'boxed', 1, a => a);
    for (const c of ['textbf', 'textit', 'texttt', 'text', 'mathrm', 'mathbf',
                     'mathcal', 'mathbb', 'mathit', 'mathsf', 'emph', 'operatorname'])
      t = expandCmd(t, c, 1, a => a);
    t = expandCmd(t, 'bar',   1, a => a + '̄');
    t = expandCmd(t, 'sqrt',  1, a => '√(' + a + ')');
    t = expandCmd(t, 'dfrac', 2, (a, b) => '(' + a + ')/(' + b + ')');
    t = expandCmd(t, 'tfrac', 2, (a, b) => '(' + a + ')/(' + b + ')');
    t = expandCmd(t, 'frac',  2, (a, b) => '(' + a + ')/(' + b + ')');
    if (t === before) break;
  }
  t = t.replace(/\\log_2/g, 'log₂').replace(/\\log/g, 'log');
  t = t.replace(/\\(min|max|sum|prod)_\{([^{}]*)\}/g,
        (m,f,sub)=> (f==='sum'?'Σ':f==='prod'?'Π':f)+'['+sub+']');
  t = t.replace(/\\sum/g,'Σ').replace(/\\nabla/g,'∇').replace(/\\infty/g,'∞');
  t = t.replace(/\\(le|leq)\b/g,'≤').replace(/\\(ge|geq)\b/g,'≥').replace(/\\neq\b/g,'≠');
  t = t.replace(/\\ll\b/g,'≪').replace(/\\gg\b/g,'≫').replace(/\\approx\b/g,'≈');
  t = t.replace(/\\times\b/g,'×').replace(/\\cdot\b/g,'·').replace(/\\pm\b/g,'±');
  t = t.replace(/\\in\b/g,'∈').replace(/\\notin\b/g,'∉').replace(/\\subset\b/g,'⊂');
  t = t.replace(/\\(Longrightarrow|Rightarrow|implies)\b/g,'⟹').replace(/\\(to|rightarrow)\b/g,'→');
  t = t.replace(/\\(ldots|dots|cdots)\b/g,'…').replace(/\\propto\b/g,'∝');
  t = t.replace(/\\gtrsim\b/g,'≳').replace(/\\lesssim\b/g,'≲');
  t = t.replace(/\\sim\b/g,'~').replace(/\\ell\b/g,'ℓ');
  t = t.replace(/\\bar\s+(\w)/g,'$1̄');
  t = t.replace(/\\mathbb\{E\}/g,'E');
  t = t.replace(/\\([A-Za-z]+)/g, (m,w)=> GREEK[w] !== undefined ? GREEK[w] : w);
  // superscripts / subscripts
  t = t.replace(/\^\{([^{}]+)\}/g, (m,g)=> [...g].every(c=>SUP[c]) ? [...g].map(c=>SUP[c]).join('') : '^('+g+')');
  t = t.replace(/\^(\w)(?![A-Za-z])/g, (m,c)=> SUP[c] || '^'+c);
  t = t.replace(/_\{([^{}]+)\}/g, (m,g)=> [...g].every(c=>SUB[c]) ? [...g].map(c=>SUB[c]).join('') : '_'+g);
  t = t.replace(/_(\w)(?![A-Za-z])/g, (m,c)=> SUB[c] || '_'+c);
  t = t.replace(/[{}]/g,'').replace(/\s+/g,' ').trim();
  return t;
}

/* ---------------- inline markdown -> runs ---------------- */
const EMPH = /\*\*\*([\s\S]+?)\*\*\*|\*\*([\s\S]+?)\*\*|\*((?:[^*]|\*\*[\s\S]*?\*\*)+?)\*(?!\*)/;

// leaf text: split out inline math and code spans
function atoms(text, base, out) {
  for (const seg of text.split(/(\$[^$]+\$|`[^`]+`)/g)) {
    if (!seg) continue;
    if (seg.startsWith('$') && seg.endsWith('$') && seg.length > 2) {
      out.push(new TextRun({ ...base, text: mathToText(seg.slice(1,-1)), font: 'Cambria Math', italics: true }));
    } else if (seg.startsWith('`') && seg.endsWith('`') && seg.length > 2) {
      out.push(new TextRun({ ...base, text: seg.slice(1,-1), font: 'Consolas', size: 18, color: ACCENT }));
    } else {
      out.push(new TextRun({ ...base, text: seg }));
    }
  }
}

// emphasis is resolved FIRST so that **bold spans containing $math$** stay intact
function emph(s, base, out, depth = 0) {
  if (depth > 4) { atoms(s, base, out); return; }
  let idx = 0, m;
  while ((m = EMPH.exec(s.slice(idx)))) {
    const at = idx + m.index;
    if (at > idx) atoms(s.slice(idx, at), base, out);
    if (m[1] !== undefined)      emph(m[1], { ...base, bold: true, italics: true }, out, depth + 1);
    else if (m[2] !== undefined) emph(m[2], { ...base, bold: true }, out, depth + 1);
    else                         emph(m[3], { ...base, italics: true }, out, depth + 1);
    idx = at + m[0].length;
  }
  if (idx < s.length) atoms(s.slice(idx), base, out);
}

function runs(text, base = {}) {
  const out = [];
  emph(text, base, out);
  return out.length ? out : [new TextRun({ ...base, text: '' })];
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 276 },
  ...opts.para,
});

/* ---------------- table builder ---------------- */
function mdTable(rows) {
  const head = rows[0], body = rows.slice(1);
  const n = head.length;
  const wgt = head.map((_, i) => {
    let m = 6;
    for (const r of rows) m = Math.max(m, Math.min((r[i] || '').length, 60));
    return m;
  });
  const tot = wgt.reduce((a, b) => a + b, 0);
  let cols = wgt.map(w => Math.round(CONTENT * w / tot));
  cols[n-1] += CONTENT - cols.reduce((a,b)=>a+b,0);

  const cell = (txt, i, isHead) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: SHADE, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({
      children: runs(txt, isHead ? { bold: true, size: 17, color: MUTED } : { size: 18 }),
      spacing: { after: 0, line: 250 },
    })],
  });

  return new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: cols,
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 4, color: 'C8CFC9' },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: 'C8CFC9' },
      left:   { style: BorderStyle.SINGLE, size: 4, color: 'C8CFC9' },
      right:  { style: BorderStyle.SINGLE, size: 4, color: 'C8CFC9' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'E2E6E2' },
      insideVertical:   { style: BorderStyle.SINGLE, size: 2, color: 'E2E6E2' },
    },
    rows: [
      new TableRow({ tableHeader: true, children: head.map((c,i)=>cell(c,i,true)) }),
      ...body.map(r => new TableRow({
        children: Array.from({length:n},(_,i)=>cell(r[i]||'',i,false)) })),
    ],
  });
}

function figure(key, caption) {
  const f = FIGS[key];
  if (!f) return [];
  const buf = fs.readFileSync(path.join(FIGDIR, path.basename(f.file)));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 60 },
      children: [new ImageRun({
        type: 'png', data: buf,
        transformation: { width: IMG_W, height: Math.round(IMG_W * f.ar) },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 220 },
      children: runs(caption, { size: 17, italics: true, color: MUTED }),
    }),
  ];
}

/* ---------------- markdown parser ---------------- */
const md = fs.readFileSync(SRC, 'utf8').split(/\r?\n/);
const els = [];
let i = 0;

const FIGMAP = {
  'Figure 1.1': ['fig1_1', 'Figure 1.1 — The synthetic urban deployment. 500 sensors over 1000×1000 m, coloured by criticality class, with buildings of varying height.'],
  'Figure 3.1': ['fig3_1', 'Figure 3.1 — (a) Adaptive descent over a cluster of high-priority sensors. (b) The planner as a deterministic coupled pipeline.'],
  'Figure 3.2': ['fig3_2', 'Figure 3.2 — Altitude against mission progress for each method.'],
  'Figure 4.1': ['fig4_1', 'Figure 4.1 — Satisfaction by criticality class. Error bars are 95% Student-t confidence intervals over twenty layouts.'],
  'Figure 4.2': ['fig4_2', 'Figure 4.2 — Cumulative distribution of achieved rate on the high-priority class. The left plateau is unserved nodes, not constraint violations.'],
  'Figure 4.3': ['fig4_3', 'Figure 4.3 — Energy decomposed into flight, hover and communication components.'],
  'Figure 4.4': ['fig4_4', 'Figure 4.4 — Distribution of hover altitudes per method.'],
  'Figure 4.5': ['fig4_5', 'Figure 4.5 — Representative trajectories over the deployment.'],
};

while (i < md.length) {
  let ln = md[i];

  // fenced code
  if (/^```/.test(ln)) {
    i++; const buf = [];
    while (i < md.length && !/^```/.test(md[i])) buf.push(md[i++]);
    i++;
    buf.forEach((c, k) => els.push(new Paragraph({
      children: [new TextRun({ text: c || ' ', font: 'Consolas', size: 17, color: '3A4441' })],
      spacing: { after: 0, before: k === 0 ? 120 : 0, line: 240 },
      shading: { type: ShadingType.CLEAR, fill: 'F2F4F1', color: 'auto' },
      indent: { left: 220, right: 220 },
    })));
    els.push(new Paragraph({ text: '', spacing: { after: 120 } }));
    continue;
  }

  // table
  if (/^\s*\|/.test(ln) && i + 1 < md.length && /^\s*\|[\s:|-]+\|\s*$/.test(md[i+1])) {
    const rows = [];
    while (i < md.length && /^\s*\|/.test(md[i])) {
      const cells = md[i].trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
      if (!/^[\s:|-]+$/.test(cells.join(''))) rows.push(cells);
      i++;
    }
    els.push(mdTable(rows));
    els.push(new Paragraph({ text: '', spacing: { after: 180 } }));
    continue;
  }

  // headings
  let m;
  if ((m = ln.match(/^(#{1,4})\s+(.*)$/))) {
    const lvl = m[1].length;
    let txt = m[2].replace(/\*\*/g, '').replace(/^\*|\*$/g, '');
    if (lvl === 1) {
      els.push(new Paragraph({ children: [new PageBreak()] }));
      els.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 60, after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 8 } },
        children: runs(txt, { bold: true, size: 34, color: INK, font: 'Georgia' }),
      }));
    } else if (lvl === 2) {
      const fk = Object.keys(FIGMAP).find(k => txt.startsWith(k));
      if (fk) els.push(new Paragraph({ children: [new PageBreak()] }));
      els.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: fk ? 0 : 320, after: 120 },
        keepNext: true,
        children: runs(txt, { bold: true, size: 26, color: ACCENT, font: 'Georgia' }),
      }));
      if (fk) figure(...FIGMAP[fk]).forEach(e => els.push(e));
    } else if (lvl === 3) {
      els.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 240, after: 100 },
        children: runs(txt, { bold: true, size: 22, color: INK }),
      }));
    } else {
      els.push(new Paragraph({
        heading: HeadingLevel.HEADING_4,
        spacing: { before: 180, after: 80 },
        children: runs(txt, { bold: true, size: 20, color: MUTED }),
      }));
    }
    i++; continue;
  }

  // display math
  if (/^\s*\$\$/.test(ln)) {
    const buf = [];
    let s = ln.replace(/^\s*\$\$/, '');
    if (/\$\$\s*$/.test(s)) { buf.push(s.replace(/\$\$\s*$/, '')); i++; }
    else {
      buf.push(s); i++;
      while (i < md.length && !/\$\$/.test(md[i])) buf.push(md[i++]);
      if (i < md.length) { buf.push(md[i].replace(/\$\$.*$/, '')); i++; }
    }
    els.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 180 },
      children: [new TextRun({ text: mathToText(buf.join(' ')), font: 'Cambria Math', size: 24, color: INK })],
    }));
    continue;
  }

  // hr
  if (/^\s*---\s*$/.test(ln)) {
    els.push(new Paragraph({
      text: '', spacing: { before: 120, after: 200 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'D6DBD7', space: 1 } },
    }));
    i++; continue;
  }

  // blockquote
  if (/^\s*>/.test(ln)) {
    const buf = [];
    while (i < md.length && /^\s*>/.test(md[i])) { buf.push(md[i].replace(/^\s*>\s?/, '')); i++; }
    const qPara = (body, isItem) => new Paragraph({
      children: runs(body, { size: 20, color: '2E3835' }),
      spacing: { before: isItem ? 20 : 100, after: isItem ? 20 : 140, line: 280 },
      indent: { left: isItem ? 560 : 320, right: 200, hanging: isItem ? 180 : 0 },
      shading: { type: ShadingType.CLEAR, fill: SHADE2, color: 'auto' },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 10 } },
    });
    for (const chunk of buf.join('\n').split(/\n\s*\n/)) {
      const lines = chunk.split('\n');
      let acc = [];
      const flush = () => { const b = acc.join(' ').trim(); if (b) els.push(qPara(b, false)); acc = []; };
      for (const l of lines) {
        const li = l.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
        if (li) { flush(); els.push(qPara('•  ' + li[1], true)); }
        else acc.push(l);
      }
      flush();
    }
    continue;
  }

  // lists
  if (/^\s*([-*]|\d+\.)\s+/.test(ln)) {
    while (i < md.length && /^\s*([-*]|\d+\.)\s+/.test(md[i])) {
      const raw = md[i];
      const indent = (raw.match(/^\s*/)[0].length >= 2) ? 1 : 0;
      const ordered = /^\s*\d+\./.test(raw);
      const txt = raw.replace(/^\s*([-*]|\d+\.)\s+/, '');
      els.push(new Paragraph({
        children: runs(txt),
        numbering: { reference: ordered ? 'numlist' : 'bullets', level: indent },
        spacing: { after: 70, line: 272 },
      }));
      i++;
    }
    continue;
  }

  // blank
  if (!ln.trim()) { i++; continue; }

  // paragraph
  const buf = [ln]; i++;
  while (i < md.length && md[i].trim() && !/^(\s*[-*]|\s*\d+\.|#{1,4}\s|>|\s*\||```|\s*\$\$|\s*---\s*$)/.test(md[i])) buf.push(md[i++]);
  els.push(P(buf.join(' ')));
}

/* ---------------- front matter ---------------- */
const rule = (color = ACCENT, size = 14) => new Paragraph({
  text: '', spacing: { after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size, color, space: 1 } },
});

const front = [
  new Paragraph({ text: '', spacing: { after: 1400 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [new TextRun({ text: 'BTP VIVA & PRESENTATION', bold: true, size: 22,
      color: ACCENT, font: 'Consolas', characterSpacing: 60 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 260 },
    children: [new TextRun({ text: 'Complete Study Guide', bold: true, size: 60,
      color: INK, font: 'Georgia' })],
  }),
  rule('C8CFC9', 8),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: 'Criticality-Aware Three-Dimensional UAV Trajectory Planning',
      size: 26, color: '2E3835', font: 'Georgia', italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 700 },
    children: [new TextRun({ text: 'for Energy-Budgeted IoT Data Collection',
      size: 26, color: '2E3835', font: 'Georgia', italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: 'Arkadip Siddha  ·  Tarun Rai  ·  Tushar Jawale  ·  Tushar Tiwari',
      size: 20, color: MUTED })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: 'Supervisor: Dr. Vivek Kumar Singh', size: 20, color: MUTED })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 900 },
    children: [new TextRun({ text: 'ABV-IIITM Gwalior  ·  Department of Information Technology',
      size: 20, color: MUTED })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Includes all eight report figures with explanations and ~93 viva questions',
      size: 19, color: ACCENT, italics: true })],
  }),

  // --- naming key page ---
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ALERT, space: 8 } },
    children: [new TextRun({ text: 'READ THIS FIRST — Figure Legend Key', bold: true,
      size: 34, color: ALERT, font: 'Georgia' })],
  }),
  P('The figure legends use the code\'s internal names, but the report text uses different names for the same four methods. If an examiner points at a slide and asks "what is Strong-Coupled?", you must answer instantly.',
    { run: { size: 21 } }),
  mdTable([
    ['Name in the FIGURES', 'Name in the REPORT TEXT', 'What it is'],
    ['2D-AUTO', 'Planar baseline', 'The fixed-altitude starting point. Cruises 46–50 m. Serves 0.6% of critical sensors.'],
    ['Two-Stage (decoupled)', 'Decoupled 3D baseline', 'The real opponent. Same altitude freedom; places hovers first, repairs rate violations after.'],
    ['Coupled-Greedy (ablation)', 'Coupled-greedy', 'OUR method with Stage 3 removed. An ablation, NOT an independent baseline.'],
    ['Strong-Coupled (deterministic)', 'Proposed method', 'OURS — the full four-stage planner with continuous altitude refinement.'],
  ]),
  new Paragraph({ text: '', spacing: { after: 240 } }),
  new Paragraph({
    children: runs('Also note: some figures carry the older project name **ATOM-3D-VoI**. That is this project. If asked, say it was the working name during development; the report presents it simply as the proposed method.',
      { size: 20, color: '2E3835' }),
    spacing: { before: 100, after: 140, line: 280 },
    indent: { left: 320, right: 200 },
    shading: { type: ShadingType.CLEAR, fill: 'F7E7DF', color: 'auto' },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: ALERT, space: 10 } },
  }),

  // --- TOC ---
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 8 } },
    children: [new TextRun({ text: 'Contents', bold: true, size: 34, color: INK, font: 'Georgia' })],
  }),
  new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-2' }),
  P('(In Word: right-click the table above and choose "Update Field" to populate page numbers.)',
    { run: { size: 17, italics: true, color: MUTED } }),
];

/* ---------------- document ---------------- */
const doc = new Document({
  creator: 'BTP Study Guide',
  title: 'BTP Viva & Presentation Study Guide',
  styles: {
    default: { document: { run: { font: 'Calibri', size: 21, color: INK }, paragraph: { spacing: { line: 276 } } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', quickFormat: true, run: { font: 'Georgia', size: 34, bold: true, color: INK } },
      { id: 'Heading2', name: 'Heading 2', quickFormat: true, run: { font: 'Georgia', size: 26, bold: true, color: ACCENT } },
      { id: 'Heading3', name: 'Heading 3', quickFormat: true, run: { size: 22, bold: true, color: INK } },
      { id: 'Heading4', name: 'Heading 4', quickFormat: true, run: { size: 20, bold: true, color: MUTED } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [
        { level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 380, hanging: 200 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '\u25E6', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 740, hanging: 200 } } } },
      ]},
      { reference: 'numlist', levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 380, hanging: 220 } } } },
        { level: 1, format: LevelFormat.LOWER_LETTER, text: '%2.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 740, hanging: 220 } } } },
      ]},
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: 15840 },
        margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
      },
    },
    children: [...front, ...els],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(OUT, b);
  console.log('wrote', OUT, (b.length / 1024 / 1024).toFixed(2), 'MB;', els.length, 'body elements');
});
