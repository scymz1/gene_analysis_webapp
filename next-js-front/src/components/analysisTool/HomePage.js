'use client';
// import { useState } from 'react';
// import ModelTrainingCard from './ModelTrainingCard';

// const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://scdrugmap.com';
export default function HomePage() {
    

    return (
        <div className="flex flex-col h-full">
            {/* Hero Section */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6 flex-shrink-0">
                <div className="p-8 bg-gradient-to-r from-blue-50 to-white">
                    <h1 className="text-4xl font-bold text-gray-800 mb-4">Welcome to scDrugMap</h1>
                    <p className="text-lg text-gray-600 max-w-2xl">
                        Your comprehensive platform for single-cell drug response analysis and mapping.
                    </p>
                </div>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {/* Analysis Tools Card */}
                <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                    <div className="flex items-center mb-4">
                        <div className="p-2 bg-blue-100 rounded-lg">
                            <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <h3 className="ml-3 text-xl font-semibold text-gray-800">Analysis Tools</h3>
                    </div>
                    <p className="text-gray-600">Powerful tools for analyzing single-cell data and drug responses.</p>
                </div>

                {/* Data Browser Card */}
                <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                    <div className="flex items-center mb-4">
                        <div className="p-2 bg-green-100 rounded-lg">
                            <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7zm0 5h16" />
                            </svg>
                        </div>
                        <h3 className="ml-3 text-xl font-semibold text-gray-800">Data Browser</h3>
                    </div>
                    <p className="text-gray-600">Browse and download comprehensive datasets and analysis results.</p>
                </div>

                {/* Visualization Card */}
                <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                    <div className="flex items-center mb-4">
                        <div className="p-2 bg-purple-100 rounded-lg">
                            <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" />
                            </svg>
                        </div>
                        <h3 className="ml-3 text-xl font-semibold text-gray-800">Visualization</h3>
                    </div>
                    <p className="text-gray-600">Interactive visualizations to explore and understand your data.</p>
                </div>
            </div>

            {/* Quick Start Guide */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">Quick Start Guide</h2>
                <div className="flex flex-col space-y-4">
                    <div className="flex items-center">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold">1</div>
                        <p className="ml-4 text-gray-600">Upload your single-cell data</p>
                    </div>
                    <div className="flex items-center">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold">2</div>
                        <p className="ml-4 text-gray-600">Select analysis parameters</p>
                    </div>
                    <div className="flex items-center">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold">3</div>
                        <p className="ml-4 text-gray-600">Explore results and visualizations</p>
                    </div>
                </div>
            </div>
        </div>
    );
} 