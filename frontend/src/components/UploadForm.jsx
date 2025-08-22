import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import toast from 'react-hot-toast';
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';

const UploadForm = () => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size must be less than 10MB');
      return;
    }

    await uploadFile(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const uploadFile = async (file) => {
    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });

      const { job_id } = response.data;
      
      toast.success('File uploaded successfully! Processing started.');
      
      // Navigate to results page
      navigate(`/results/${job_id}`);
      
    } catch (error) {
      console.error('Upload failed:', error);
      const errorMessage = error.response?.data?.detail || 'Upload failed. Please try again.';
      toast.error(errorMessage);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const getFileIcon = (file) => {
    const extension = file.name.split('.').pop().toLowerCase();
    const iconClass = "h-8 w-8 text-maritime-blue";
    
    if (extension === 'pdf') {
      return <DocumentIcon className={iconClass} />;
    } else if (['docx', 'doc'].includes(extension)) {
      return <DocumentIcon className={iconClass} />;
    } else if (['png', 'jpg', 'jpeg', 'tiff'].includes(extension)) {
      return <DocumentIcon className={iconClass} />;
    }
    
    return <DocumentIcon className={iconClass} />;
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card-large">
        <div className="text-center mb-8">
          <div className="mx-auto h-16 w-16 bg-maritime-blue bg-opacity-10 rounded-full flex items-center justify-center mb-4">
            <CloudArrowUpIcon className="h-8 w-8 text-maritime-blue" />
          </div>
          <h2 className="text-2xl font-bold text-maritime-navy mb-2">
            Upload Statement of Facts
          </h2>
          <p className="text-maritime-gray-600">
            Upload your maritime document to extract port events automatically
          </p>
        </div>

        {/* Upload Zone */}
        <div
          {...getRootProps()}
          className={`
            upload-zone cursor-pointer transition-all duration-300
            ${isDragActive ? 'upload-zone-active scale-102' : ''}
            ${uploading ? 'pointer-events-none opacity-60' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          {uploading ? (
            <div className="space-y-4">
              <div className="spinner mx-auto"></div>
              <div>
                <p className="text-maritime-gray-600 mb-2">
                  Uploading and processing...
                </p>
                <div className="w-full bg-maritime-gray-200 rounded-full h-2">
                  <div
                    className="bg-maritime-blue h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-sm text-maritime-gray-500 mt-1">
                  {uploadProgress}% complete
                </p>
              </div>
            </div>
          ) : acceptedFiles.length > 0 ? (
            <div className="space-y-4">
              <CheckCircleIcon className="h-12 w-12 text-green-500 mx-auto" />
              <div className="flex items-center justify-center space-x-3">
                {getFileIcon(acceptedFiles[0])}
                <div>
                  <p className="font-medium text-maritime-navy">
                    {acceptedFiles[0].name}
                  </p>
                  <p className="text-sm text-maritime-gray-500">
                    {(acceptedFiles[0].size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <p className="text-maritime-gray-600">
                Click to upload or drop another file
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <CloudArrowUpIcon className="h-12 w-12 text-maritime-gray-400 mx-auto" />
              {isDragActive ? (
                <p className="text-maritime-blue font-medium">
                  Drop the file here...
                </p>
              ) : (
                <div>
                  <p className="text-maritime-gray-600 mb-2">
                    Drag & drop your file here, or{' '}
                    <span className="text-maritime-blue font-medium">browse</span>
                  </p>
                  <p className="text-sm text-maritime-gray-500">
                    Supports PDF, DOCX, DOC, and image files (PNG, JPG, TIFF)
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Supported Formats */}
        <div className="mt-8 p-4 bg-maritime-gray-50 rounded-lg">
          <h3 className="font-medium text-maritime-navy mb-2 flex items-center">
            <ExclamationTriangleIcon className="h-4 w-4 mr-2" />
            Supported Formats
          </h3>
          <div className="grid grid-cols-2 gap-4 text-sm text-maritime-gray-600">
            <div>
              <p className="font-medium mb-1">Documents</p>
              <ul className="space-y-1">
                <li>• PDF files</li>
                <li>• Microsoft Word (.docx, .doc)</li>
              </ul>
            </div>
            <div>
              <p className="font-medium mb-1">Images (OCR)</p>
              <ul className="space-y-1">
                <li>• PNG, JPG, JPEG</li>
                <li>• TIFF files</li>
              </ul>
            </div>
          </div>
          <p className="text-xs text-maritime-gray-500 mt-3">
            Maximum file size: 10MB
          </p>
        </div>
      </div>
    </div>
  );
};

export default UploadForm;
