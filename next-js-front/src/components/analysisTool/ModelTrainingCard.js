import { useState } from 'react';
import FinetuneButton from './FinetuneButton';
import FixedEmbeddingButton from './FixedEmbeddingButton';

export default function ModelTrainingCard({ selectedModel }) {
    const [activeTab, setActiveTab] = useState('fixed');

    return (
        <div className="mt-8 flex-1">
            {/* Model Selection Section */}
            {/* <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
                <div className="p-6 bg-gradient-to-r from-blue-50 to-white">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">
                        Step 1: Select a Model
                    </h3>
                    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                        {modelOptions.map((model) => (
                            <button
                                key={model}
                                onClick={() => setSelectedModel(model)}
                                className={`p-2.5 rounded-md border transition-all duration-200 
                                    ${selectedModel === model 
                                        ? 'border-blue-500 bg-blue-50 shadow-sm' 
                                        : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/50'}
                                    text-center`}
                            >
                                <span className="text-sm font-medium text-gray-700">
                                    {model}
                                </span>
                            </button>
                        ))}
                    </div>
                </div>
            </div> */}

            {/* Training Options Section - Only shown after model selection */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden transition-all duration-300 h-full">
                <div className="p-6 border-b">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">
                        Step 2: Choose Training Method
                    </h3>
                    <div className="flex border rounded-lg overflow-hidden">
                        <button
                            onClick={() => setActiveTab('fixed')}
                            className={`flex-1 py-4 px-6 text-sm font-medium transition-colors duration-200
                                ${activeTab === 'fixed'
                                    ? 'text-blue-600 bg-blue-50'
                                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                                }`}
                        >
                            <div className="flex items-center justify-center space-x-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                                            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                </svg>
                                <span>Fixed Embeddings</span>
                            </div>
                        </button>
                        <button
                            onClick={() => setActiveTab('finetune')}
                            className={`flex-1 py-4 px-6 text-sm font-medium transition-colors duration-200
                                ${activeTab === 'finetune'
                                    ? 'text-blue-600 bg-blue-50'
                                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                                }`}
                        >
                            <div className="flex items-center justify-center space-x-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                                            d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2M7 7h10" />
                                </svg>
                                <span>Model Finetuning</span>
                            </div>
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6">
                    <div className={`${activeTab === 'fixed' ? 'block' : 'hidden'}`}>
                        <FixedEmbeddingButton selectedModel={selectedModel} />
                    </div>
                    <div className={`${activeTab === 'finetune' ? 'block' : 'hidden'}`}>
                        <FinetuneButton selectedModel={selectedModel} />
                    </div>
                </div>
            </div>

        </div>
    );
} 