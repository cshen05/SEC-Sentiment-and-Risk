CREATE TABLE IF NOT EXISTS filings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    form_type VARCHAR(10) NOT NULL,
    filing_date DATE,
    accession_number VARCHAR(30) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paragraphs (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    section VARCHAR(50),
    text TEXT NOT NULL,
    weak_label VARCHAR(20),
    manual_label VARCHAR(20),
    prediction VARCHAR(20),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS unique_paragraph_per_filing
ON paragraphs (filing_id, section, text);