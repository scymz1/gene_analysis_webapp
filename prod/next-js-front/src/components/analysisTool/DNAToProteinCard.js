"use client";

import { useState } from 'react';

export default function DNAToProteinCard() {
  // React state to handle the input value and selected genome
  const [dnaInput, setDnaInput] = useState('');
  const [selectedGenome, setSelectedGenome] = useState('HG38');

  // Predefined example DNA substitutions
  const exampleDNA =
    'chr17:g.7673776G>A,chr17:g.7673776C>A,chr11:g.6622666G>A,chr15:g.75688603A>G,chr2:g.21006635G>T,chr3:g.169381339G>T,chr6:g.158980908A>C,chr6:17799397T>C,chr7:g.140753336A>T';

  const handleLoadExample = () => {
    setDnaInput(exampleDNA); // Set the textarea with the example DNA
    setSelectedGenome('HG38'); // Select HG38 as the default reference genome
  };

  return (
    <div className="h-full flex flex-col">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">DNA To Protein Conversion</h2>

      <div className="flex flex-col flex-grow">
        <label
          htmlFor="dna-input"
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300 flex items-center"
        >
          Enter DNA substitutions for conversion.{' '}
          <span
            className="ml-2 inline-block w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            title="Provide substitutions in the format: chr#:g.###>[Sub]"
          >
            ❔
          </span>
        </label>

        <textarea
          id="dna-input"
          rows="5"
          value={dnaInput} // Controlled component
          onChange={(e) => setDnaInput(e.target.value)} // Update state on change
          className="block w-full p-3 mb-4 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white"
          placeholder="chr17:g.7673776G>A, chr11:g.6622666G>A..."
        ></textarea>

        <fieldset className="mb-4">
          <legend className="text-sm font-medium text-gray-900 dark:text-gray-300 mb-2">
            Reference Genome
          </legend>
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="radio"
                name="reference-genome"
                value="HG19"
                checked={selectedGenome === 'HG19'} // Controlled component
                onChange={(e) => setSelectedGenome(e.target.value)} // Update state
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600"
              />
              <span className="ml-2 text-sm text-gray-900 dark:text-gray-300">HG19</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="reference-genome"
                value="HG38"
                checked={selectedGenome === 'HG38'} // Controlled component
                onChange={(e) => setSelectedGenome(e.target.value)} // Update state
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600"
              />
              <span className="ml-2 text-sm text-gray-900 dark:text-gray-300">HG38</span>
            </label>
          </div>
        </fieldset>

        <div className="flex space-x-4">
          <button
            type="button"
            onClick={handleLoadExample} // Load example on click
            className="px-4 py-2 bg-teal-500 text-white text-sm font-medium rounded-lg shadow hover:bg-teal-600 focus:ring-4 focus:outline-none focus:ring-teal-300 dark:bg-teal-700 dark:hover:bg-teal-800 dark:focus:ring-teal-800"
          >
            Load Example
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-orange-500 text-white text-sm font-medium rounded-lg shadow hover:bg-orange-600 focus:ring-4 focus:outline-none focus:ring-orange-300 dark:bg-orange-700 dark:hover:bg-orange-800 dark:focus:ring-orange-800"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}
