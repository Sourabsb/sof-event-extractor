# SoF Event Extractor - Sample Document Templates

This directory contains sample Statement of Facts documents for testing the SoF Event Extractor system.

## Sample Event Types to Include in Test Documents:

### Port Operations
- Vessel arrival at port
- Vessel departure from port  
- Berthing/unberthing operations
- Anchor dropping/weighing

### Cargo Operations
- Loading commencement/completion
- Discharge commencement/completion
- Cargo inspection events
- Hold cleaning operations

### Maritime Services
- Pilot boarding/disembarking
- Tug assistance start/end
- Bunker operations
- Fresh water supply

### Regulatory & Documentation
- Port clearance obtained
- Customs inspection
- Immigration clearance
- Health inspection

### Weather & Delays
- Weather delays
- Equipment breakdowns
- Waiting for berth availability
- Traffic delays

## Sample Timestamp Formats to Test:

- DD/MM/YYYY HH:MM
- MM/DD/YYYY HH:MM  
- YYYY-MM-DD HH:MM
- DD-MM-YYYY HH:MM
- Various time zones (UTC, local time)

## Test Document Formats:

1. **Clean PDF** - Computer-generated PDF with selectable text
2. **Scanned PDF** - Image-based PDF requiring OCR
3. **Word Document** - DOCX format with tables and formatting
4. **Image Files** - PNG/JPG scans of handwritten or printed documents

## Usage:

Place your test documents in this directory and use them to validate:
- Text extraction accuracy
- Event identification precision  
- Timestamp parsing reliability
- Location/port name extraction
- Duration calculation accuracy

The system should achieve 95%+ accuracy on well-formatted maritime documents.
