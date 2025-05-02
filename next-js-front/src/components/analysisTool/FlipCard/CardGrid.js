"use client";
import HoverFlipCard from "./HoverFlipCard";
import { useRouter } from 'next/navigation';

export default function CardGrid() {
  const router = useRouter();

  const tools = [
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            Data Preparation
          </h3>
          {/* <p className="mt-2">🧬</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Please read the instructions before using the tools.
        </p>
      ),
      color: "#38b2ac", // Teal
      link: "/instructions"
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            Tools
          </h3>
          {/* <p className="mt-2">🔄</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Tools for predicting RNA secondary structures.
        </p>
      ),
      color: "#d69e2e", // Yellow
      link: "/Data"
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
          Explore Analysis
          </h3>
          {/* <p className="mt-2">🧪</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Explore and download analysis results.
        </p>
      ),
      color: "#9f7aea", // Purple
      link: "/Data"
    },
    // {
    //   frontContent: (
    //     <div>
    //       <h3 className="text-lg font-bold uppercase text-center">
    //         GENE NAME NORMALIZER
    //       </h3>
    //       {/* <p className="mt-2">🌿</p> */}
    //     </div>
    //   ),
    //   backContent: (
    //     <p className="text-center">
    //       Normalize gene names across various databases.
    //     </p>
    //   ),
    //   color: "#48bb78", // Green
    // },
    // {
    //   frontContent: (
    //     <div>
    //       <h3 className="text-lg font-bold uppercase text-center">
    //         REVERSE COMPLEMENT
    //       </h3>
    //       {/* <p className="mt-2">🔁</p> */}
    //     </div>
    //   ),
    //   backContent: (
    //     <p className="text-center">
    //       Generate the reverse complement of a DNA sequence.
    //     </p>
    //   ),
    //   color: "#f56565", // Red
    // },
  ];

  return (
    <div className="h-full flex flex-col gap-4">
      <br></br>
      {tools.map((tool, index) => (
        <div key={index} onClick={() => router.push(tool.link)} className="cursor-pointer">
          <HoverFlipCard
            frontContent={tool.frontContent}
            backContent={tool.backContent}
            color={tool.color}
          />
        </div>
      ))}
    </div>
  );
}
