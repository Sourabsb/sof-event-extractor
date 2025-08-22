import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { HomeIcon, ClockIcon, InformationCircleIcon } from '@heroicons/react/24/outline';

const Navbar = () => {
  const location = useLocation();

  const navigation = [
    { name: 'Upload', href: '/', icon: HomeIcon },
    { name: 'History', href: '/history', icon: ClockIcon },
    { name: 'About', href: '/about', icon: InformationCircleIcon },
  ];

  return (
    <nav className="bg-maritime-navy border-b border-maritime-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="h-8 w-8 bg-maritime-blue rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">⚓</span>
            </div>
            <div className="flex flex-col">
              <span className="text-white font-semibold text-lg">SoF Extractor</span>
              <span className="text-maritime-blue text-xs">Maritime AI</span>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex space-x-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              const Icon = item.icon;
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`
                    flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200
                    ${isActive
                      ? 'bg-maritime-blue text-white'
                      : 'text-maritime-gray-300 hover:text-white hover:bg-maritime-gray-800'
                    }
                  `}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-maritime-gray-300 text-sm">API Online</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
