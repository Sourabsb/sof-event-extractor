import React from 'react';
import { 
  ShieldCheckIcon,
  CpuChipIcon,
  GlobeAltIcon,
  ChartBarIcon,
  DocumentTextIcon,
  CloudArrowUpIcon,
  AcademicCapIcon,
  BuildingOffice2Icon
} from '@heroicons/react/24/outline';

const About = () => {
  const features = [
    {
      icon: CpuChipIcon,
      title: 'Advanced AI Processing',
      description: 'Combines regex patterns, spaCy NLP, and GPT models for accurate event extraction from complex maritime documents.'
    },
    {
      icon: DocumentTextIcon,
      title: 'Multi-Format Support',
      description: 'Processes PDF, DOCX, DOC files and scanned images using OCR technology with Azure Cognitive Services.'
    },
    {
      icon: ChartBarIcon,
      title: 'Visual Analytics',
      description: 'Interactive timeline visualizations and statistical analysis of extracted port events and operational data.'
    },
    {
      icon: CloudArrowUpIcon,
      title: 'Secure Processing',
      description: 'Enterprise-grade security with encrypted file uploads and secure data processing pipelines.'
    },
    {
      icon: GlobeAltIcon,
      title: 'Maritime Expertise',
      description: 'Built specifically for maritime industry needs with understanding of port operations and shipping terminology.'
    },
    {
      icon: ShieldCheckIcon,
      title: 'High Accuracy',
      description: '95%+ accuracy rate in extracting port events, timestamps, and location data from Statement of Facts documents.'
    }
  ];

  const stats = [
    { label: 'Document Types', value: '5+', description: 'PDF, DOCX, Images' },
    { label: 'Processing Speed', value: '<30s', description: 'Average processing time' },
    { label: 'Accuracy Rate', value: '95%+', description: 'Event extraction accuracy' },
    { label: 'Languages', value: '10+', description: 'OCR language support' }
  ];

  const techStack = [
    { category: 'Frontend', technologies: ['React 18', 'TailwindCSS', 'Recharts', 'React Router'] },
    { category: 'Backend', technologies: ['FastAPI', 'Python 3.8+', 'Pandas', 'Asyncio'] },
    { category: 'AI/ML', technologies: ['OpenAI GPT', 'spaCy NLP', 'Azure Cognitive', 'Pytesseract'] },
    { category: 'Processing', technologies: ['PyPDF2', 'python-docx', 'Pillow', 'Regex'] }
  ];

  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center">
        <div className="mx-auto h-16 w-16 bg-maritime-navy rounded-2xl flex items-center justify-center mb-6">
          <span className="text-white text-2xl">⚓</span>
        </div>
        <h1 className="text-4xl font-bold text-maritime-gradient mb-4">
          About SoF Event Extractor
        </h1>
        <p className="text-xl text-maritime-gray-600 max-w-3xl mx-auto">
          Revolutionizing maritime document processing with AI-powered extraction technology 
          designed specifically for the shipping industry's operational needs.
        </p>
      </div>

      {/* Mission Statement */}
      <div className="card-large text-center">
        <AcademicCapIcon className="h-12 w-12 text-maritime-blue mx-auto mb-6" />
        <h2 className="text-2xl font-bold text-maritime-navy mb-4">Our Mission</h2>
        <p className="text-lg text-maritime-gray-600 max-w-4xl mx-auto leading-relaxed">
          To eliminate manual processing bottlenecks in maritime operations by providing 
          intelligent, accurate, and scalable document analysis tools that transform 
          unstructured Statement of Facts into actionable business intelligence.
        </p>
      </div>

      {/* Problem & Solution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <div className="h-12 w-12 bg-red-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-red-600 text-2xl">⚠️</span>
          </div>
          <h3 className="text-xl font-semibold text-maritime-navy mb-3">The Problem</h3>
          <ul className="space-y-2 text-maritime-gray-600">
            <li>• Manual extraction takes 2-4 hours per document</li>
            <li>• High error rates due to human fatigue</li>
            <li>• Inconsistent data formats across operators</li>
            <li>• Difficulty scaling operations globally</li>
            <li>• Limited integration with existing systems</li>
          </ul>
        </div>

        <div className="card">
          <div className="h-12 w-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-green-600 text-2xl">✅</span>
          </div>
          <h3 className="text-xl font-semibold text-maritime-navy mb-3">Our Solution</h3>
          <ul className="space-y-2 text-maritime-gray-600">
            <li>• Automated extraction in under 30 seconds</li>
            <li>• 95%+ accuracy with AI validation</li>
            <li>• Standardized JSON/CSV output formats</li>
            <li>• RESTful API for easy integration</li>
            <li>• Supports multiple document formats</li>
          </ul>
        </div>
      </div>

      {/* Features Grid */}
      <div>
        <h2 className="text-2xl font-bold text-maritime-navy text-center mb-8">
          Key Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div key={index} className="card">
                <div className="h-12 w-12 bg-maritime-blue bg-opacity-10 rounded-lg flex items-center justify-center mb-4">
                  <Icon className="h-6 w-6 text-maritime-blue" />
                </div>
                <h3 className="text-lg font-semibold text-maritime-navy mb-2">
                  {feature.title}
                </h3>
                <p className="text-maritime-gray-600 text-sm">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Statistics */}
      <div className="card-large">
        <h2 className="text-2xl font-bold text-maritime-navy text-center mb-8">
          System Performance
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-maritime-blue mb-2">
                {stat.value}
              </div>
              <div className="font-semibold text-maritime-navy mb-1">
                {stat.label}
              </div>
              <div className="text-sm text-maritime-gray-500">
                {stat.description}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Technology Stack */}
      <div>
        <h2 className="text-2xl font-bold text-maritime-navy text-center mb-8">
          Technology Stack
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {techStack.map((stack, index) => (
            <div key={index} className="card">
              <h3 className="font-semibold text-maritime-navy mb-3">
                {stack.category}
              </h3>
              <ul className="space-y-1">
                {stack.technologies.map((tech, techIndex) => (
                  <li key={techIndex} className="text-sm text-maritime-gray-600">
                    • {tech}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture Overview */}
      <div className="card-large">
        <h2 className="text-2xl font-bold text-maritime-navy text-center mb-6">
          System Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-blue rounded-full flex items-center justify-center mx-auto mb-4">
              <CloudArrowUpIcon className="h-8 w-8 text-white" />
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">Frontend Layer</h3>
            <p className="text-sm text-maritime-gray-600">
              React-based UI with maritime-themed design, drag-drop uploads, 
              and real-time processing status updates.
            </p>
          </div>
          
          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-blue rounded-full flex items-center justify-center mx-auto mb-4">
              <CpuChipIcon className="h-8 w-8 text-white" />
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">Processing Engine</h3>
            <p className="text-sm text-maritime-gray-600">
              FastAPI backend with async processing, multi-format parsers, 
              and intelligent AI extraction pipeline.
            </p>
          </div>
          
          <div className="text-center">
            <div className="h-16 w-16 bg-maritime-blue rounded-full flex items-center justify-center mx-auto mb-4">
              <ChartBarIcon className="h-8 w-8 text-white" />
            </div>
            <h3 className="font-semibold text-maritime-navy mb-2">Data Layer</h3>
            <p className="text-sm text-maritime-gray-600">
              Structured data export, timeline visualization, 
              and RESTful API for system integration.
            </p>
          </div>
        </div>
      </div>

      {/* IME Integration */}
      <div className="card-large text-center wave-animation">
        <BuildingOffice2Icon className="h-12 w-12 text-white mx-auto mb-6" />
        <h2 className="text-2xl font-bold text-white mb-4">
          Built for Integrated Maritime Exchange
        </h2>
        <p className="text-blue-100 max-w-3xl mx-auto leading-relaxed">
          This solution is designed as a hackathon submission for IME, demonstrating 
          how AI can revolutionize maritime document processing and enhance operational 
          efficiency across the shipping industry. Ready for integration with existing 
          maritime management systems.
        </p>
      </div>

      {/* Contact/Support */}
      <div className="card text-center">
        <h3 className="text-xl font-semibold text-maritime-navy mb-4">
          Questions or Support?
        </h3>
        <p className="text-maritime-gray-600 mb-6">
          Get in touch with our team for technical support, integration assistance, 
          or to discuss enterprise deployment options.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <button className="btn-secondary">
            Technical Documentation
          </button>
          <button className="btn-primary">
            Contact Team
          </button>
        </div>
      </div>
    </div>
  );
};

export default About;
