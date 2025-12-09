'use client';

import Image from 'next/image';

export default function ReadmePage() {
    return (
        <div className="flex flex-col h-full p-6">
            <div className="bg-white rounded-2xl shadow-lg p-8 flex flex-col lg:flex-row gap-8">
                {/* Left: Textual Description */}
                <div className="flex-1">
                    <h1 className="text-3xl font-bold text-gray-800 mb-4">
                        About <span className="text-blue-600">scDrugMap</span>
                    </h1>
                    <p className="text-gray-700 mb-4">
                        <strong>scDrugMap</strong> is a scalable and user-friendly framework for predicting drug responses in single-cell transcriptomics data using large-scale foundation models. It supports both a Python command-line tool and an interactive web interface (<a href="https://scdrugmap.com" className="text-blue-600 underline" target="_blank">scdrugmap.com</a>) to facilitate drug discovery and translational research.
                    </p>

                    <h2 className="text-xl font-semibold text-gray-800 mb-2">Overview</h2>
                    <p className="text-gray-700 mb-4">
                        Drug resistance remains a major challenge in cancer treatment. Single-cell profiling provides critical insights into cellular heterogeneity and drug sensitivity, but high dimensionality and data sparsity can complicate downstream analysis.
                    </p>

                    <p className="text-gray-700 mb-4">
                        <strong>scDrugMap</strong> evaluates and integrates the predictive power of eight single-cell foundation models and two natural language models across diverse cancer types and treatment regimens. It supports:
                    </p>

                    <ul className="list-disc list-inside text-gray-700 space-y-1">
                        <li>Pooled-data and cross-data model evaluation</li>
                        <li>Layer freezing and LoRA fine-tuning</li>
                        <li>Zero-shot and fine-tuned inference</li>
                        <li>Single-cell resolution predictions across 326,000+ cells and 36+ datasets</li>
                    </ul>
                </div>

                {/* Right: PNG Image */}
                <div className="flex-1 flex justify-center items-center">
                    <Image
                        src="/readme_fig.jpg"
                        alt="scDrugMap Structure Overview"
                        width={900}
                        height={800}
                        className="rounded-lg shadow-md max-w-full h-auto"
                        priority
                    />
                </div>
            </div>

            {/* Specific Instructions Section */}
            <div className="bg-white rounded-2xl shadow-lg p-8 mt-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Specific Instructions</h2>

                <h3 className="text-lg font-semibold text-gray-800 mt-4">1. Select LLM Model</h3>
                <p className="text-gray-700 mb-4">
                    Choose the large language model (LLM) you want to use for drug response prediction from the dropdown menu.
                </p>
                <Image src="/instruction_step1.png" alt="Select LLM Model" width={700} height={400} className="rounded-lg shadow-md mb-6" />

                <h3 className="text-lg font-semibold text-gray-800 mt-4">2. Upload CSV/TSV File</h3>
                <p className="text-gray-700 mb-4">
                    Upload a tab-delimited gene expression matrix (.tsv, .csv, .txt) for single cells:
                </p>
                <ul className="list-disc list-inside text-gray-700 mb-4">
                    <li>Each row = one single cell</li>
                    <li><strong>Cell_barcode</strong>: Unique cell ID (e.g., C70R_C70R.bcDWVD)</li>
                    <li><strong>Condition</strong>: Experimental condition (e.g., resistant, sensitive)</li>
                    <li>Remaining columns: Gene symbols (e.g., AAAS, AACS, …) with integer counts</li>
                </ul>
                <p className="text-gray-700 mb-4">
                    The example input file can be downloaded and used for testing purpose.
                </p>
                <Image src="/instruction_step2.png" alt="Upload File" width={700} height={400} className="rounded-lg shadow-md mb-6" />

                <h3 className="text-lg font-semibold text-gray-800 mt-4">3. Upload & Preprocess Data</h3>
                <p className="text-gray-700 mb-4">
                    After selecting your file, click the <strong>Upload & Preprocess Data</strong> button. The system will automatically preprocess the data for analysis.
                </p>
                <Image src="/instruction_step3.png" alt="Upload and Preprocess" width={700} height={400} className="rounded-lg shadow-md mb-6" />

                <h3 className="text-lg font-semibold text-gray-800 mt-4">4. Choose Prediction Mode</h3>
                <p className="text-gray-700 mb-4">
                    You can either:
                </p>
                <ul className="list-disc list-inside text-gray-700 mb-4">
                    <li><strong>Fixed embeddings</strong>: Fast prediction of accuracy, precision, and F1 score.</li>
                    <li><strong>Fine-tuning</strong>: Train the model on your uploaded data for better performance (slower).</li>
                </ul>
                <p className="text-gray-700 mb-4">
                    Adjustable parameters:
                </p>
                <ul className="list-disc list-inside text-gray-700 mb-4">
                    <li>Number of Epochs</li>
                    <li>Training Rate</li>
                    <li>Learning Rate</li>
                </ul>
                <Image src="/instruction_step4.png" alt="Upload and Preprocess" width={700} height={400} className="rounded-lg shadow-md mb-6" />
                <p className="text-gray-700 mb-4">
                    The training progress, loss curves, and metrics are displayed in real time.
                </p>
                <Image src="/instruction_step5.png" alt="Training Progress" width={700} height={400} className="rounded-lg shadow-md mb-6" />

                <h3 className="text-lg font-semibold text-gray-800 mt-4">5. Download Trained Model</h3>
                <p className="text-gray-700 mb-4">
                    Once training is complete, click <strong>Download Model</strong> to save the fine-tuned model to your local machine, also <strong>Download Test Records</strong> to save the test records to your local machine.
                </p>
                <Image src="/instruction_step6.png" alt="Download Model" width={700} height={400} className="rounded-lg shadow-md mb-6" />
            </div>
        </div>
    );
}
