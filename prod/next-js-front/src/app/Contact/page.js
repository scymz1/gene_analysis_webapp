'use client';

import React from 'react';
// import Image from 'next/image';

export default function Contact() {
  return (
    <div className="min-h-screen pt-20 px-4 md:px-8">
      <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold mb-8">Contact</h1>

        {/* Citation Section */}
        {/* <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 text-orange-600">Citation of the Tool</h2>
          <p className="text-gray-700 mb-4">To cite PhenoProfiler in a publication, please quote the following:</p>
          <p className="text-gray-700 bg-gray-50 p-4 rounded">
            &ldquo;Bo L, Bob Z, Chengyang Z, Song Q et al. PhenoProfiler : Advancing Morphology Representations for 
            Image-based Drug Discovery ...&rdquo;
          </p>
        </div> */}
{/* Citation Section */}
	<div className="mb-12">
	  <h2 className="text-xl font-semibold mb-4 text-orange-600">Citation of the Paper</h2>
	  <p className="text-gray-700 mb-4">
	    To cite <span className="italic">scDrugMap</span>, please use the following reference:
	  </p>
	  <blockquote className="border-l-4 border-orange-500 pl-4 italic text-gray-800 bg-gray-50 p-4 rounded-md shadow-sm">
	    Wang Q, Pan Y, <strong>Zhou M</strong>, Tang Z, Wang Y, Wang G, Song Q. <span className="italic">scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction</span>. arXiv preprint arXiv:2505.05612. 2025 May 8.
	  </blockquote>
	</div>
        {/* Contact Info Section */}
        <div className="mb-8">

	  <h2 className="text-xl font-semibold mb-4 text-orange-600">Contact Information</h2>
          
          <div className="space-y-4">
            {/* Primary Contact */}
            <div>
              <p className="font-semibold">Qianqian Song, PhD</p>
              <p className="text-gray-700">Email: qsong1@ufl.edu</p>
            </div>

            {/* <div>
              <p className="font-semibold">Bob Zhang, PhD</p>
              <p className="text-gray-700">Email: bobzhang@um.edu.mo</p>
            </div>

            <div>
              <p className="font-semibold">Chengyang Zhang, PhD</p>
              <p className="text-gray-700">Email: Cy_Zhang0705@163.com</p>
            </div>

            <div>
              <p className="font-semibold">Bo Li, PhD</p>
              <p className="text-gray-700">Email: 19919920960@163.com</p>
            </div>

            <div>
              <p className="font-semibold">Minghao Zhou, MSc</p>
              <p className="text-gray-700">Email: minghao.zhou@ufl.edu</p>
            </div> */}

            {/* Mailing Address */}
            <div className="mt-4">
              <p className="font-semibold">Mailing address:</p>
              <div className="text-gray-700">
                <p>Department of health outcomes and biomedical informatics</p>
                <p>College of Medicine</p>
                <p>University of Florida</p>
                <p>1889 Museum Rd, Suite 7000, Gainesville, FL 32611</p>
                {/* <p>Tel: (352) 627-9467</p> */}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 pt-4 border-t text-sm text-gray-600">
          <div className="flex justify-between items-center">
            <div>
              <p>Copyright 2025-Present - University of Florida</p>
              <div className="flex gap-4 mt-2">
                <a href="#" className="text-blue-600 hover:underline">Emergency Information</a>
                <span>|</span>
                <a href="#" className="text-blue-600 hover:underline">Site Policies</a>
              </div>
            </div>
            <div>
              <img 
                src="/UF_logo.png" 
                alt="University of Florida Logo" 
                className="h-10 object-contain"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 
