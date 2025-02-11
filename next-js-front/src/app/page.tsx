'use client';

import { useState } from 'react';
import Image from "next/image";

export default function Home() {
  const [resultFile, setResultFile] = useState<string | null>(null);

  const handleAnalysis = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData();
    
    const fileInput = e.currentTarget.querySelector('input[type="file"]') as HTMLInputElement;
    console.log(fileInput);
    if (fileInput?.files && fileInput.files.length === 5) {
      Array.from(fileInput.files).forEach((file, index) => {
        formData.append(`image_${index}`, file);  // Give each file a unique key
      });

      try {
        const response = await fetch('https://phenoprofiler.org/phenoProfiler/analyze/', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          setResultFile(url);
        } else {
          const errorData = await response.json();
          console.error('Analysis failed:', errorData);
          alert(errorData.error || 'Analysis failed');
        }
      } catch (error) {
        console.error('Error:', error);
        alert('Error processing images');
      }
    } else {
      alert('Please upload exactly 5 channel images');
    }
  };

  const handleDownload = () => {
    if (resultFile) {
      const link = document.createElement('a');
      link.href = resultFile;
      link.download = 'morphology_profiles.npy';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Center Container with Side Lines */}
      <div className="max-w-5xl mx-auto min-h-screen relative bg-white shadow-2xl">
        {/* Decorative Side Lines */}
        <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b  shadow-lg"></div>
        <div className="absolute inset-y-0 right-0 w-1 bg-gradient-to-b  shadow-lg"></div>

        {/* Header with Logo and Title */}
        <header className="bg-gradient-to-b from-white to-gray-50 p-6 border-b border-gray-200 shadow-sm">
          <div className="flex items-center gap-4">
            {/* <Image
              src="/logo.svg"
              alt="Gene Analysis Logo"
              width={80}
              height={80}
            /> */}
            <h1 className="text-2xl text-blue-800 font-semibold">
              Comprehensive reference map of drug resistance mechanisms in human cancer
            </h1>
          </div>
        </header>

        {/* Decorative Divider */}
        <hr className="border-gray-200" />

        {/* Main Content */}
        <main className="p-6 flex-grow bg-white">
          <div className="space-y-8">
            {/* Rest of your content sections... */}
            {/* Each section gets additional styling */}
            <div className="bg-white rounded-lg p-6 shadow-sm hover:shadow-md transition duration-200">
              <h2 className="text-xl font-bold mb-6">About PhenoProfiler</h2>
              
              {/* About Section - Two Column Layout */}
              <div className="flex gap-8">
                {/* Left Column - Image */}
                <div className="w-1/2">
                  <Image
                    src="/PhenoProfiler_structure.png"
                    alt="PhenoProfiler Structure"
                    width={500}
                    height={400}
                    className="w-full h-auto object-contain"
                  />
                </div>

                {/* Right Column - Text */}
                <div className="w-1/2 prose max-w-none">
                  <p className="mb-4">
                    <span className="font-bold">PhenoProfiler</span> is an advanced tool for phenotypic profiling of cell morphology, 
                    to efficiently extract phenotypic effects of perturbations from high-throughput imaging.
                  </p>
                  <p className="mb-8">
                    <span className="font-bold">PhenoProfiler</span> operates as an end-to-end image encoder, converting multi-channel 
                    images directly into low-dimensional quantitative features, thus eliminating the 
                    need for extensive preprocessing in non-end-to-end pipeline. For more details, please refer to our {' '}
                    <a 
                      href="https://github.com/QSong-github/PhenoProfiler" 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="text-blue-600 hover:text-blue-800 underline"
                    >
                      github repository
                    </a>.
                  </p>
                </div>
              </div>
            </div>

            <hr className="border-gray-200" />

            {/* Image Analysis Section */}
            <div className="bg-white rounded-lg p-6 shadow-sm hover:shadow-md transition duration-200">
              <h3 className="text-lg font-semibold mb-4">Generate Morphology Profiles</h3>
              <form onSubmit={handleAnalysis}>
                <div className="space-y-6">
                  {/* Image Upload */}
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                    <label className="block mb-2 font-medium">Upload Cell Images (PNG format)</label>
                    <div className="flex flex-col items-center justify-center">
                      <input
                        type="file"
                        accept=".png"
                        multiple  // Allow multiple files
                        className="block w-full text-sm text-gray-500
                          file:mr-4 file:py-2 file:px-4
                          file:rounded-md file:border-0
                          file:text-sm file:font-semibold
                          file:bg-blue-50 file:text-blue-700
                          hover:file:bg-blue-100"
                      />
                      <p className="mt-2 text-sm text-gray-500">
                        Supported format: PNG images (Please upload 5 channel images)
                      </p>
                    </div>
                  </div>

                  {/* Analysis Button */}
                  <div className="flex justify-center">
                    <button 
                      type="submit"
                      className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700
                                flex items-center gap-2 font-medium"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                              d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Generate Morphology Profiles
                    </button>
                  </div>

                  {/* Results Section */}
                  {resultFile && (
                    <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium mb-3">Analysis Results</h4>
                      <div className="flex items-center justify-between p-3 bg-white rounded border">
                        <span className="text-sm">morphology_profiles.npy</span>
                        <button 
                          onClick={handleDownload}
                          className="px-4 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700
                                    flex items-center gap-2"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                          Download Results
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </form>
            </div>

            <hr className="border-gray-200" />

            {/* Additional sections with same styling pattern */}
            <div className="bg-white rounded-lg p-6 shadow-sm hover:shadow-md transition duration-200">
              {/* ... other sections ... */}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
