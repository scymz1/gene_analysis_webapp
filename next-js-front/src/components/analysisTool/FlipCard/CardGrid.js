import HoverFlipCard from "./HoverFlipCard";

export default function CardGrid() {
  const tools = [
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            DNA TO PROTEIN CONVERSION
          </h3>
          {/* <p className="mt-2">🧬</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Convert a genomic sequence into its protein sequence.
        </p>
      ),
      color: "#38b2ac", // Teal
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            GENOMIC LIFTOVER
          </h3>
          {/* <p className="mt-2">🔄</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Convert genomic coordinates between assemblies.
        </p>
      ),
      color: "#d69e2e", // Yellow
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            PROTEIN TO DNA CONVERSION
          </h3>
          {/* <p className="mt-2">🧪</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Convert protein sequences back to DNA sequences.
        </p>
      ),
      color: "#9f7aea", // Purple
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            GENE NAME NORMALIZER
          </h3>
          {/* <p className="mt-2">🌿</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Normalize gene names across various databases.
        </p>
      ),
      color: "#48bb78", // Green
    },
    {
      frontContent: (
        <div>
          <h3 className="text-lg font-bold uppercase text-center">
            REVERSE COMPLEMENT
          </h3>
          {/* <p className="mt-2">🔁</p> */}
        </div>
      ),
      backContent: (
        <p className="text-center">
          Generate the reverse complement of a DNA sequence.
        </p>
      ),
      color: "#f56565", // Red
    },
  ];

  return (
    <div className="h-full flex flex-col gap-4">
      {tools.map((tool, index) => (
        <HoverFlipCard
          key={index}
          frontContent={tool.frontContent}
          backContent={tool.backContent}
          color={tool.color}
        />
      ))}
    </div>
  );
}
