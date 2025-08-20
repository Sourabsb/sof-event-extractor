import React from 'react';
import UploadForm from '../components/UploadForm';
import { 
  DocumentTextIcon,
  CloudArrowUpIcon,
  ChartBarIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline';

const Home = () => {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center py-12">
        <div className="mx-auto h-20 w-20 bg-maritime-navy rounded-2xl flex items-center justify-center mb-6">
          <span className="text-white text-3xl">⚓</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-maritime-gradient mb-6">
          SoF Event Extractor
        </h1>
        <p className="text-xl text-maritime-gray-600 max-w-3xl mx-auto mb-8">
          Transform your maritime Statement of Facts documents into structured data using 
          advanced AI technology. Extract port events, timestamps, and locations automatically.
        </p>
        <div className="flex flex-wrap justify-center gap-4 text-sm text-maritime-gray-500">
          <div className="flex items-center">
            <CloudArrowUpIcon className="h-5 w-5 mr-2 text-maritime-blue" />
            PDF & DOCX Support
          </div>
          <div className="flex items-center">
            <DocumentTextIcon className="h-5 w-5 mr-2 text-maritime-blue" />
            OCR for Scanned Images
          </div>
          <div className="flex items-center">
            <ChartBarIcon className="h-5 w-5 mr-2 text-maritime-blue" />
            Timeline Visualization
          </div>
          <div className="flex items-center">
            <ArrowDownTrayIcon className="h-5 w-5 mr-2 text-maritime-blue" />
            CSV & JSON Export
          </div>
        </div>
      </div>

      {/* Upload Form */}
      <UploadForm />

      {/* Features Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="card text-center">
          <div className="h-12 w-12 bg-maritime-blue bg-opacity-10 rounded-lg flex items-center justify-center mx-auto mb-4">
            <DocumentTextIcon className="h-6 w-6 text-maritime-blue" />
          </div>
          <h3 className="text-lg font-semibold text-maritime-navy mb-2">
            Smart Document Processing
          </h3>
          <p className="text-maritime-gray-600">
            Automatically extract text from PDFs, Word documents, and scanned images 
            using advanced OCR technology.
          </p>
        </div>

        <div className="card text-center">
          <div className="h-12 w-12 bg-maritime-blue bg-opacity-10 rounded-lg flex items-center justify-center mx-auto mb-4">
            <ChartBarIcon className="h-6 w-6 text-maritime-blue" />
          </div>
          <h3 className="text-lg font-semibold text-maritime-navy mb-2">
            AI-Powered Extraction
          </h3>
          <p className="text-maritime-gray-600">
            Uses NLP and GPT models to intelligently identify and extract port events, 
            timestamps, and location data.
          </p>
        </div>

        <div className="card text-center">
          <div className="h-12 w-12 bg-maritime-blue bg-opacity-10 rounded-lg flex items-center justify-center mx-auto mb-4">
            <ArrowDownTrayIcon className="h-6 w-6 text-maritime-blue" />
          </div>
          <h3 className="text-lg font-semibold text-maritime-navy mb-2">
            Flexible Export Options
          </h3>
          <p className="text-maritime-gray-600">
            Export your structured data as CSV or JSON files for integration 
            with existing maritime systems.
          </p>
        </div>
      </div>

      {/* How it Works */}
      <div className="card-large">
        <h2 className="text-2xl font-bold text-maritime-navy text-center mb-8">
          How It Works
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-navy rounded-full flex items-center justify-center mx-auto mb-4 text-white font-bold text-xl">
              1
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">Upload Document</h3>
            <p className="text-sm text-maritime-gray-600">
              Upload your SoF document in PDF, DOCX, or image format
            </p>
          </div>

          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-navy rounded-full flex items-center justify-center mx-auto mb-4 text-white font-bold text-xl">
              2
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">AI Processing</h3>
            <p className="text-sm text-maritime-gray-600">
              Our AI extracts text and identifies maritime events and timestamps
            </p>
          </div>

          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-navy rounded-full flex items-center justify-center mx-auto mb-4 text-white font-bold text-xl">
              3
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">Review Results</h3>
            <p className="text-sm text-maritime-gray-600">
              View extracted events in a structured table with timeline visualization
            </p>
          </div>

          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-navy rounded-full flex items-center justify-center mx-auto mb-4 text-white font-bold text-xl">
              4
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">Export Data</h3>
            <p className="text-sm text-maritime-gray-600">
              Download your structured data as CSV or JSON for further use
            </p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        <div className="text-center">
          <div className="text-3xl font-bold text-maritime-navy">95%</div>
          <div className="text-sm text-maritime-gray-600">Accuracy Rate</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-maritime-navy">&lt;30s</div>
          <div className="text-sm text-maritime-gray-600">Processing Time</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-maritime-navy">5+</div>
          <div className="text-sm text-maritime-gray-600">File Formats</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-maritime-navy">24/7</div>
          <div className="text-sm text-maritime-gray-600">Available</div>
        </div>
      </div>
    </div>
  );
};

export default Home;
