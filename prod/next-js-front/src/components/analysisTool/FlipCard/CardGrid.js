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
            README
          </h3>
          {/* <p className="mt-2">🧬</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Read the README to prepare your data before using LLM tools.
        </p>
      ),
      color: "#38b2ac", // Teal
      link: "/readme"
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            LLM Tools
          </h3>
          {/* <p className="mt-2">🔄</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Run LLMs tools using your single-cell data and drug responses.
        </p>
      ),
      color: "#d69e2e", // Yellow
      link: "/Data"
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            Data Browser
          </h3>
          {/* <p className="mt-2">🧪</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Browse and download related single-cell datasets.
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
