import React, { useState, useEffect } from 'react';
import { 
  CalculatorIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ArrowDownTrayIcon,
  CheckCircleIcon,
  XCircleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import axios from 'axios';

const LayTimeCalculator = ({ jobId, summary, events, onExport }) => {
  const [summaryData, setSummaryData] = useState({
    'CREATED FOR': '',
    'VOYAGE FROM': '',
    'VOYAGE TO': '',
    'CARGO': '',
    'PORT': '',
    'OPERATION': 'Discharge',
    'DEMURRAGE': 0,
    'DISPATCH': 0,
    'LOAD/DISCH': 0,
    'CARGO QTY': 0,
    ...summary
  });

  const [laytimeResult, setLaytimeResult] = useState(null);
  const [calculating, setCalculating] = useState(false);
  const [events_data, setEventsData] = useState(events || []);

  // Update events when props change
  useEffect(() => {
    if (events) {
      setEventsData(events);
    }
  }, [events]);

  const handleSummaryChange = (field, value) => {
    setSummaryData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleEventChange = (index, field, value) => {
    const updatedEvents = [...events_data];
    updatedEvents[index] = {
      ...updatedEvents[index],
      [field]: value
    };
    setEventsData(updatedEvents);
  };

  const addNewEvent = () => {
    const newEvent = {
      event: '',
      start_time_iso: new Date().toISOString(),
      end_time_iso: null,
      laytime_counts: false,
      raw_line: 'Manually added event',
      filename: 'Manual Entry'
    };
    setEventsData([...events_data, newEvent]);
  };

  const removeEvent = (index) => {
    const updatedEvents = events_data.filter((_, i) => i !== index);
    setEventsData(updatedEvents);
  };

  const calculateLaytime = async () => {
    setCalculating(true);
    try {
      // First update the events in the backend
      await axios.put(`http://localhost:8000/api/update-events/${jobId}`, events_data);

      // Then calculate laytime
      const response = await axios.post(`http://localhost:8000/api/calculate-laytime/${jobId}`, summaryData);
      setLaytimeResult(response.data);
      toast.success('Laytime calculation completed!');
    } catch (error) {
      console.error('Calculation error:', error);
      toast.error(error.response?.data?.detail || 'Calculation failed');
    } finally {
      setCalculating(false);
    }
  };

  const exportLaytime = async (format) => {
    try {
      toast.loading(`Preparing ${format.toUpperCase()} export...`);
      
      const response = await axios.post(
        `http://localhost:8000/api/export/${jobId}?export_type=${format}&data_type=laytime`,
        {},
        { responseType: 'blob' }
      );
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `laytime_calculation_${jobId.substring(0, 8)}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.dismiss();
      toast.success(`${format.toUpperCase()} file downloaded successfully!`);
      
    } catch (err) {
      toast.dismiss();
      console.error('Export failed:', err);
      toast.error(`Export failed: ${err.response?.data?.detail || 'Unknown error'}`);
    }
  };

  const formatDateTime = (dateTime) => {
    if (!dateTime) return '';
    try {
      const date = new Date(dateTime);
      return date.toISOString().slice(0, 16); // Format for datetime-local input
    } catch {
      return '';
    }
  };

  const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(num);
  };

  return (
    <div className="space-y-8">
      {/* Voyage Summary Section */}
      <div className="card">
        <h3 className="text-xl font-bold text-maritime-navy mb-6 flex items-center">
          <InformationCircleIcon className="h-6 w-6 mr-2" />
          Voyage Summary
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Created For (Vessel Name)*
              </label>
              <input
                type="text"
                value={summaryData['CREATED FOR']}
                onChange={(e) => handleSummaryChange('CREATED FOR', e.target.value)}
                className="input-field"
                placeholder="e.g., MV ALRAYAN"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Voyage From*
              </label>
              <input
                type="text"
                value={summaryData['VOYAGE FROM']}
                onChange={(e) => handleSummaryChange('VOYAGE FROM', e.target.value)}
                className="input-field"
                placeholder="e.g., Singapore"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Port*
              </label>
              <input
                type="text"
                value={summaryData['PORT']}
                onChange={(e) => handleSummaryChange('PORT', e.target.value)}
                className="input-field"
                placeholder="e.g., Port of Rotterdam"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Demurrage ($/Day)*
              </label>
              <input
                type="number"
                value={summaryData['DEMURRAGE']}
                onChange={(e) => handleSummaryChange('DEMURRAGE', parseFloat(e.target.value) || 0)}
                className="input-field"
                placeholder="e.g., 12000"
                min="0"
                step="0.01"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Load/Discharge Rate (MT/Day)*
              </label>
              <input
                type="number"
                value={summaryData['LOAD/DISCH']}
                onChange={(e) => handleSummaryChange('LOAD/DISCH', parseFloat(e.target.value) || 0)}
                className="input-field"
                placeholder="e.g., 10000"
                min="0"
                step="0.01"
              />
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Voyage To*
              </label>
              <input
                type="text"
                value={summaryData['VOYAGE TO']}
                onChange={(e) => handleSummaryChange('VOYAGE TO', e.target.value)}
                className="input-field"
                placeholder="e.g., Rotterdam"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Cargo*
              </label>
              <input
                type="text"
                value={summaryData['CARGO']}
                onChange={(e) => handleSummaryChange('CARGO', e.target.value)}
                className="input-field"
                placeholder="e.g., Coal"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Operation*
              </label>
              <select
                value={summaryData['OPERATION']}
                onChange={(e) => handleSummaryChange('OPERATION', e.target.value)}
                className="input-field"
              >
                <option value="Discharge">Discharge</option>
                <option value="Loading">Loading</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Dispatch Rate*
              </label>
              <input
                type="number"
                value={summaryData['DISPATCH']}
                onChange={(e) => handleSummaryChange('DISPATCH', parseFloat(e.target.value) || 0)}
                className="input-field"
                placeholder="e.g., 6000"
                min="0"
                step="0.01"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-maritime-gray-700 mb-2">
                Cargo Quantity (MT)*
              </label>
              <input
                type="number"
                value={summaryData['CARGO QTY']}
                onChange={(e) => handleSummaryChange('CARGO QTY', parseFloat(e.target.value) || 0)}
                className="input-field"
                placeholder="e.g., 65000"
                min="0"
                step="0.01"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Events Editor Section */}
      <div className="card">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold text-maritime-navy flex items-center">
            <ClockIcon className="h-6 w-6 mr-2" />
            Statement of Facts Timeline
          </h3>
          <button
            onClick={addNewEvent}
            className="btn-secondary"
          >
            Add Event
          </button>
        </div>
        
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {events_data.map((event, index) => (
            <div key={index} className="border border-maritime-gray-200 rounded-lg p-4 bg-maritime-gray-50">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-maritime-gray-700 mb-1">
                    Event*
                  </label>
                  <input
                    type="text"
                    value={event.event}
                    onChange={(e) => handleEventChange(index, 'event', e.target.value)}
                    className="input-field text-sm"
                    placeholder="Event description"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-maritime-gray-700 mb-1">
                    Start Time*
                  </label>
                  <input
                    type="datetime-local"
                    value={formatDateTime(event.start_time_iso)}
                    onChange={(e) => handleEventChange(index, 'start_time_iso', e.target.value ? new Date(e.target.value).toISOString() : null)}
                    className="input-field text-sm"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-maritime-gray-700 mb-1">
                    End Time
                  </label>
                  <input
                    type="datetime-local"
                    value={formatDateTime(event.end_time_iso)}
                    onChange={(e) => handleEventChange(index, 'end_time_iso', e.target.value ? new Date(e.target.value).toISOString() : null)}
                    className="input-field text-sm"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="block text-sm font-medium text-maritime-gray-700 mb-1">
                      Laytime Counts
                    </label>
                    <input
                      type="checkbox"
                      checked={event.laytime_counts || false}
                      onChange={(e) => handleEventChange(index, 'laytime_counts', e.target.checked)}
                      className="h-4 w-4 text-maritime-blue focus:ring-maritime-blue border-gray-300 rounded"
                    />
                  </div>
                  <button
                    onClick={() => removeEvent(index)}
                    className="text-red-600 hover:text-red-800 p-1"
                    title="Remove event"
                  >
                    <XCircleIcon className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Calculate Button */}
      <div className="text-center">
        <button
          onClick={calculateLaytime}
          disabled={calculating}
          className={`btn-primary text-lg px-8 py-3 ${calculating ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {calculating ? (
            <>
              <div className="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Calculating...
            </>
          ) : (
            <>
              <CalculatorIcon className="h-6 w-6 mr-2" />
              Calculate Laytime
            </>
          )}
        </button>
      </div>

      {/* Results Section */}
      {laytimeResult && (
        <div className="space-y-6">
          {/* Summary Results */}
          <div className="card">
            <h3 className="text-xl font-bold text-maritime-navy mb-6">
              Laytime Calculation Results
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {formatNumber(laytimeResult.laytime_allowed_days)} Days
                </div>
                <div className="text-sm text-blue-800">Laytime Allowed</div>
              </div>
              
              <div className="text-center p-4 bg-yellow-50 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">
                  {formatNumber(laytimeResult.laytime_consumed_days)} Days
                </div>
                <div className="text-sm text-yellow-800">Laytime Consumed</div>
              </div>
              
              <div className="text-center p-4 bg-green-50 rounded-lg">
                {laytimeResult.demurrage_due > 0 ? (
                  <>
                    <div className="text-2xl font-bold text-red-600">
                      ${formatNumber(laytimeResult.demurrage_due)}
                    </div>
                    <div className="text-sm text-red-800">Demurrage Due</div>
                  </>
                ) : laytimeResult.dispatch_due > 0 ? (
                  <>
                    <div className="text-2xl font-bold text-green-600">
                      ${formatNumber(laytimeResult.dispatch_due)}
                    </div>
                    <div className="text-sm text-green-800">Dispatch Due</div>
                  </>
                ) : (
                  <>
                    <div className="text-2xl font-bold text-green-600">
                      <CheckCircleIcon className="h-8 w-8 mx-auto" />
                    </div>
                    <div className="text-sm text-green-800">On Time</div>
                  </>
                )}
              </div>
            </div>

            {/* Export Buttons */}
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => exportLaytime('csv')}
                className="btn-secondary"
              >
                <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                Download CSV
              </button>
              <button
                onClick={() => exportLaytime('json')}
                className="btn-primary"
              >
                <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                Download JSON
              </button>
            </div>
          </div>

          {/* Calculation Log */}
          <div className="card">
            <h4 className="text-lg font-semibold text-maritime-navy mb-4">
              Calculation Log
            </h4>
            <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
              {laytimeResult.calculation_log?.map((entry, index) => (
                <div key={index} className="text-sm text-gray-700 font-mono mb-1">
                  {entry}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LayTimeCalculator;
