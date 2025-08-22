import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Timeline } from 'recharts';
import { format, parseISO, isValid } from 'date-fns';

const TimelineVisualization = ({ events, onClose }) => {
  const timelineData = useMemo(() => {
    if (!events || events.length === 0) return [];

    // Process events for timeline visualization
    const processedEvents = events
      .filter(event => event.start)
      .map((event, index) => {
        let startDate;
        let endDate;
        
        try {
          startDate = new Date(event.start);
          if (!isValid(startDate)) throw new Error('Invalid start date');
        } catch {
          return null;
        }

        try {
          endDate = event.end ? new Date(event.end) : startDate;
          if (!isValid(endDate)) endDate = startDate;
        } catch {
          endDate = startDate;
        }

        const duration = endDate > startDate ? (endDate - startDate) / (1000 * 60 * 60) : 0; // hours

        return {
          id: index,
          event: event.event,
          start: startDate,
          end: endDate,
          startTime: startDate.getTime(),
          endTime: endDate.getTime(),
          duration: Math.max(duration, 0.5), // Minimum 30 minutes for visualization
          location: event.location || 'Unknown',
          description: event.description || '',
          formattedStart: format(startDate, 'MMM dd, HH:mm'),
          formattedEnd: format(endDate, 'MMM dd, HH:mm'),
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.startTime - b.startTime);

    return processedEvents;
  }, [events]);

  // Create chart data for event frequency by hour
  const hourlyData = useMemo(() => {
    const hourCounts = {};
    
    timelineData.forEach(event => {
      const hour = event.start.getHours();
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;
    });

    return Array.from({ length: 24 }, (_, hour) => ({
      hour: `${hour.toString().padStart(2, '0')}:00`,
      events: hourCounts[hour] || 0,
    }));
  }, [timelineData]);

  const getEventColor = (eventType) => {
    const type = eventType.toLowerCase();
    if (type.includes('arrival') || type.includes('arrived')) return '#10B981'; // green
    if (type.includes('departure') || type.includes('departed')) return '#3B82F6'; // blue
    if (type.includes('loading') || type.includes('discharge')) return '#8B5CF6'; // purple
    if (type.includes('anchor')) return '#F59E0B'; // yellow
    if (type.includes('pilot') || type.includes('tug')) return '#6366F1'; // indigo
    return '#6B7280'; // gray
  };

  if (!timelineData.length) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-xl max-w-2xl w-full p-8 text-center">
          <h3 className="text-xl font-semibold text-maritime-navy mb-4">
            No Timeline Data Available
          </h3>
          <p className="text-maritime-gray-600 mb-6">
            No events with valid timestamps were found for timeline visualization.
          </p>
          <button onClick={onClose} className="btn-primary">
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-maritime-gray-200">
          <div className="flex justify-between items-center">
            <h3 className="text-xl font-semibold text-maritime-navy">
              Event Timeline Visualization
            </h3>
            <button
              onClick={onClose}
              className="text-maritime-gray-500 hover:text-maritime-gray-700"
            >
              <span className="sr-only">Close</span>
              ✕
            </button>
          </div>
        </div>

        <div className="p-6 space-y-8">
          {/* Timeline Chart */}
          <div>
            <h4 className="text-lg font-medium text-maritime-navy mb-4">
              Events Timeline
            </h4>
            <div className="bg-maritime-gray-50 rounded-lg p-4">
              <div className="space-y-4">
                {timelineData.map((event, index) => (
                  <div
                    key={event.id}
                    className="flex items-center space-x-4 p-3 bg-white rounded-lg shadow-sm"
                  >
                    <div
                      className="w-4 h-4 rounded-full flex-shrink-0"
                      style={{ backgroundColor: getEventColor(event.event) }}
                    ></div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <h5 className="font-medium text-maritime-navy truncate">
                          {event.event}
                        </h5>
                        <span className="text-sm text-maritime-gray-500">
                          {event.duration > 0 ? `${event.duration.toFixed(1)}h` : 'Instant'}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-sm text-maritime-gray-600">
                        <span>{event.formattedStart}</span>
                        {event.duration > 0 && (
                          <>
                            <span>→</span>
                            <span>{event.formattedEnd}</span>
                          </>
                        )}
                        {event.location && (
                          <>
                            <span>•</span>
                            <span>{event.location}</span>
                          </>
                        )}
                      </div>
                      {event.description && (
                        <p className="text-sm text-maritime-gray-500 mt-1 truncate">
                          {event.description}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Hourly Distribution Chart */}
          <div>
            <h4 className="text-lg font-medium text-maritime-navy mb-4">
              Event Distribution by Hour
            </h4>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hourlyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey="hour" 
                    stroke="#6B7280"
                    fontSize={12}
                  />
                  <YAxis stroke="#6B7280" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#001F3F',
                      border: 'none',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                    formatter={(value) => [value, 'Events']}
                  />
                  <Bar 
                    dataKey="events" 
                    fill="#0074D9" 
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Summary Statistics */}
          <div>
            <h4 className="text-lg font-medium text-maritime-navy mb-4">
              Timeline Summary
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-maritime-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-maritime-navy">
                  {timelineData.length}
                </div>
                <div className="text-sm text-maritime-gray-600">Total Events</div>
              </div>
              <div className="bg-maritime-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-maritime-navy">
                  {timelineData.filter(e => e.duration > 0).length}
                </div>
                <div className="text-sm text-maritime-gray-600">With Duration</div>
              </div>
              <div className="bg-maritime-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-maritime-navy">
                  {Math.round(timelineData.reduce((sum, e) => sum + e.duration, 0))}h
                </div>
                <div className="text-sm text-maritime-gray-600">Total Duration</div>
              </div>
              <div className="bg-maritime-gray-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-maritime-navy">
                  {new Set(timelineData.map(e => e.location)).size}
                </div>
                <div className="text-sm text-maritime-gray-600">Locations</div>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-maritime-gray-200">
          <div className="flex justify-end">
            <button onClick={onClose} className="btn-primary">
              Close Timeline
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TimelineVisualization;
