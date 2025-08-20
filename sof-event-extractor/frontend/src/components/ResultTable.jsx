import React, { useState } from 'react';
import { format } from 'date-fns';
import { 
  CalendarIcon, 
  MapPinIcon,
  ClockIcon,
  DocumentTextIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  ChartBarIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

const ResultTable = ({ events, jobId, onExport, onViewTimeline }) => {
  const [sortField, setSortField] = useState('start');
  const [sortDirection, setSortDirection] = useState('asc');
  const [filterLocation, setFilterLocation] = useState('');
  const [filterEvent, setFilterEvent] = useState('');

  if (!events || events.length === 0) {
    return (
      <div className="card text-center py-12">
        <DocumentTextIcon className="h-12 w-12 text-maritime-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-maritime-gray-900 mb-2">
          No Events Found
        </h3>
        <p className="text-maritime-gray-600">
          No port events were extracted from the document.
        </p>
      </div>
    );
  }

  // Check for validation warnings
  const hasWarning = events.some(event => 
    event.event === 'Document Validation Warning' || 
    event.severity === 'Warning'
  );

  if (hasWarning) {
    const warningEvent = events.find(event => 
      event.event === 'Document Validation Warning' || 
      event.severity === 'Warning'
    );
    
    return (
      <div className="space-y-6">
        {/* Warning Card */}
        <div className="card border-amber-200 bg-amber-50">
          <div className="flex items-start space-x-3">
            <ExclamationTriangleIcon className="h-8 w-8 text-amber-500 flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-amber-800 mb-2">
                Document Validation Warning
              </h3>
              <p className="text-amber-700 mb-4">
                {warningEvent.description}
              </p>
              {warningEvent.suggestion && (
                <div className="bg-white/70 rounded-lg p-3 border-l-4 border-amber-400">
                  <p className="text-sm text-amber-800">
                    <strong>Suggestion:</strong> {warningEvent.suggestion}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Requirements Information */}
        <div className="card bg-blue-50 border-blue-200">
          <h4 className="font-semibold text-blue-900 mb-3">Required Document Format:</h4>
          <ul className="space-y-2 text-blue-800 text-sm">
            <li className="flex items-center space-x-2">
              <CalendarIcon className="h-4 w-4" />
              <span>Event timestamps (start and/or end times)</span>
            </li>
            <li className="flex items-center space-x-2">
              <MapPinIcon className="h-4 w-4" />
              <span>Location information (ports, coordinates)</span>
            </li>
            <li className="flex items-center space-x-2">
              <DocumentTextIcon className="h-4 w-4" />
              <span>Maritime event descriptions (incidents, arrivals, departures)</span>
            </li>
          </ul>
        </div>
      </div>
    );
  }

  // Sort and filter events
  const processedEvents = events
    .filter(event => {
      const locationMatch = !filterLocation || 
        (event.location && event.location.toLowerCase().includes(filterLocation.toLowerCase()));
      const eventMatch = !filterEvent || 
        event.event.toLowerCase().includes(filterEvent.toLowerCase());
      return locationMatch && eventMatch;
    })
    .sort((a, b) => {
      let aValue = a[sortField] || '';
      let bValue = b[sortField] || '';
      
      if (sortField === 'start' || sortField === 'end') {
        aValue = new Date(aValue || '1900-01-01');
        bValue = new Date(bValue || '1900-01-01');
      }
      
      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const formatDateTime = (dateTime) => {
    if (!dateTime) return 'Not specified';
    try {
      const date = new Date(dateTime);
      return format(date, 'MMM dd, yyyy HH:mm');
    } catch {
      return dateTime;
    }
  };

  const getEventTypeColor = (eventType) => {
    const type = eventType.toLowerCase();
    if (type.includes('arrival') || type.includes('arrived')) {
      return 'bg-green-100 text-green-800';
    } else if (type.includes('departure') || type.includes('departed')) {
      return 'bg-blue-100 text-blue-800';
    } else if (type.includes('loading') || type.includes('discharge')) {
      return 'bg-purple-100 text-purple-800';
    } else if (type.includes('anchor')) {
      return 'bg-yellow-100 text-yellow-800';
    } else if (type.includes('pilot') || type.includes('tug')) {
      return 'bg-indigo-100 text-indigo-800';
    }
    return 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Header with actions */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-maritime-navy">
            Extracted Events
          </h2>
          <p className="text-maritime-gray-600">
            {processedEvents.length} events found in the document
          </p>
        </div>
        
        <div className="flex flex-wrap gap-2">
          {onViewTimeline && (
            <button
              onClick={onViewTimeline}
              className="btn-secondary"
            >
              <ChartBarIcon className="h-4 w-4 mr-2" />
              Timeline View
            </button>
          )}
          
          <button
            onClick={() => onExport('csv')}
            className="btn-secondary"
          >
            <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
            Export CSV
          </button>
          
          <button
            onClick={() => onExport('json')}
            className="btn-primary"
          >
            <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
              Filter by Location
            </label>
            <input
              type="text"
              placeholder="Enter location..."
              value={filterLocation}
              onChange={(e) => setFilterLocation(e.target.value)}
              className="input-field"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
              Filter by Event Type
            </label>
            <input
              type="text"
              placeholder="Enter event type..."
              value={filterEvent}
              onChange={(e) => setFilterEvent(e.target.value)}
              className="input-field"
            />
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-maritime-gray-200">
            <thead className="bg-maritime-gray-50">
              <tr>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider cursor-pointer hover:text-maritime-navy"
                  onClick={() => handleSort('event')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Event</span>
                    {sortField === 'event' && (
                      <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider cursor-pointer hover:text-maritime-navy"
                  onClick={() => handleSort('start')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Start Time</span>
                    {sortField === 'start' && (
                      <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider cursor-pointer hover:text-maritime-navy"
                  onClick={() => handleSort('end')}
                >
                  <div className="flex items-center space-x-1">
                    <span>End Time</span>
                    {sortField === 'end' && (
                      <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider cursor-pointer hover:text-maritime-navy"
                  onClick={() => handleSort('location')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Location</span>
                    {sortField === 'location' && (
                      <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Duration
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-maritime-gray-500 uppercase tracking-wider">
                  Description
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-maritime-gray-200">
              {processedEvents.map((event, index) => {
                const duration = event.start && event.end
                  ? (() => {
                      try {
                        const startDate = new Date(event.start);
                        const endDate = new Date(event.end);
                        const diffMs = endDate - startDate;
                        const hours = Math.floor(diffMs / (1000 * 60 * 60));
                        const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
                        return `${hours}h ${minutes}m`;
                      } catch {
                        return 'N/A';
                      }
                    })()
                  : 'N/A';

                return (
                  <tr key={index} className="table-stripe">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <span className={`
                          inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                          ${getEventTypeColor(event.event)}
                        `}>
                          {event.event}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-maritime-gray-900">
                      <div className="flex items-center">
                        <CalendarIcon className="h-4 w-4 text-maritime-gray-400 mr-2" />
                        {formatDateTime(event.start)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-maritime-gray-900">
                      <div className="flex items-center">
                        <CalendarIcon className="h-4 w-4 text-maritime-gray-400 mr-2" />
                        {formatDateTime(event.end)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-maritime-gray-900">
                      <div className="flex items-center">
                        <MapPinIcon className="h-4 w-4 text-maritime-gray-400 mr-2" />
                        {event.location || 'Not specified'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-maritime-gray-900">
                      <div className="flex items-center">
                        <ClockIcon className="h-4 w-4 text-maritime-gray-400 mr-2" />
                        {duration}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-maritime-gray-900">
                      <div className="max-w-xs truncate" title={event.description}>
                        {event.description || 'No description'}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <div className="text-2xl font-bold text-maritime-navy">
            {processedEvents.length}
          </div>
          <div className="text-sm text-maritime-gray-600">Total Events</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-maritime-navy">
            {new Set(processedEvents.map(e => e.location).filter(Boolean)).size}
          </div>
          <div className="text-sm text-maritime-gray-600">Unique Locations</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-maritime-navy">
            {processedEvents.filter(e => e.start && e.end).length}
          </div>
          <div className="text-sm text-maritime-gray-600">With Duration</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-maritime-navy">
            {new Set(processedEvents.map(e => e.event.toLowerCase())).size}
          </div>
          <div className="text-sm text-maritime-gray-600">Event Types</div>
        </div>
      </div>
    </div>
  );
};

export default ResultTable;
