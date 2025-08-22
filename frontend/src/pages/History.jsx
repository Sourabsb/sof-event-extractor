import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import { 
  ClockIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { useNavigate } from 'react-router-dom';

const History = () => {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchJobHistory();
  }, []);

  const fetchJobHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/api/jobs');
      setJobs(response.data.jobs || []);
    } catch (err) {
      console.error('Error fetching job history:', err);
      setError('Failed to fetch job history');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'processing':
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <ExclamationCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusBadge = (status) => {
    const baseClasses = 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium';
    switch (status) {
      case 'processing':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'completed':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'failed':
        return `${baseClasses} bg-red-100 text-red-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM dd, yyyy HH:mm');
    } catch {
      return dateString;
    }
  };

  const handleViewJob = (jobId) => {
    navigate(`/results/${jobId}`);
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="spinner mx-auto mb-4"></div>
        <h2 className="text-xl font-semibold text-maritime-navy mb-2">
          Loading History...
        </h2>
        <p className="text-maritime-gray-600">
          Fetching your processing history.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <ExclamationCircleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-maritime-navy mb-2">
          Error Loading History
        </h2>
        <p className="text-maritime-gray-600 mb-4">
          {error}
        </p>
        <button
          onClick={fetchJobHistory}
          className="btn-primary"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!jobs.length) {
    return (
      <div className="text-center py-12">
        <DocumentTextIcon className="h-12 w-12 text-maritime-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-maritime-navy mb-2">
          No Processing History
        </h2>
        <p className="text-maritime-gray-600 mb-6">
          You haven't processed any documents yet. Upload your first Statement of Facts to get started.
        </p>
        <button
          onClick={() => navigate('/')}
          className="btn-primary"
        >
          Upload Document
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-maritime-navy">
            Processing History
          </h1>
          <p className="text-maritime-gray-600 mt-2">
            View and manage your document processing history
          </p>
        </div>
        <button
          onClick={() => navigate('/')}
          className="btn-primary"
        >
          Upload New Document
        </button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="card text-center">
          <div className="text-2xl font-bold text-maritime-navy">
            {jobs.length}
          </div>
          <div className="text-sm text-maritime-gray-600">Total Jobs</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-green-600">
            {jobs.filter(job => job.status === 'completed').length}
          </div>
          <div className="text-sm text-maritime-gray-600">Completed</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-yellow-600">
            {jobs.filter(job => job.status === 'processing').length}
          </div>
          <div className="text-sm text-maritime-gray-600">Processing</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-red-600">
            {jobs.filter(job => job.status === 'failed').length}
          </div>
          <div className="text-sm text-maritime-gray-600">Failed</div>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-maritime-gray-200">
            <thead className="bg-maritime-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Document
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Job ID
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-maritime-gray-200">
              {jobs.map((job, index) => (
                <tr key={job.job_id} className="table-stripe hover:bg-maritime-gray-25">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <DocumentTextIcon className="h-5 w-5 text-maritime-gray-400 mr-3 flex-shrink-0" />
                      <div className="text-sm font-medium text-maritime-gray-900 truncate max-w-xs" title={job.filename}>
                        {job.filename}
                      </div>
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {getStatusIcon(job.status)}
                      <span className={`ml-2 ${getStatusBadge(job.status)} capitalize`}>
                        {job.status}
                      </span>
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-maritime-gray-900">
                    <div className="flex items-center">
                      <ClockIcon className="h-4 w-4 text-maritime-gray-400 mr-2" />
                      {formatDate(job.created_at)}
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-maritime-gray-900 font-mono">
                    {job.job_id.substring(0, 8)}...
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex justify-end space-x-2">
                      <button
                        onClick={() => handleViewJob(job.job_id)}
                        className="text-maritime-blue hover:text-maritime-navy"
                        title="View Details"
                      >
                        <EyeIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="text-center">
        <button
          onClick={fetchJobHistory}
          className="btn-secondary"
        >
          Refresh History
        </button>
      </div>
    </div>
  );
};

export default History;
