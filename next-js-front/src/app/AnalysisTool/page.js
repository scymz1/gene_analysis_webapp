// import DNAToProteinCard from "@/components/analysisTool/DNAToProteinCard";
import CardGrid from "../../components/analysisTool/FlipCard/CardGrid";
import CSVUploadCard from "@/components/analysisTool/CSVUploadCard";

export default function AnalysisTool() {
  return (
    <div className="p-10 min-h-screen">
      <div className="flex gap-8 h-[calc(100vh-theme(spacing.20))]">
        {/* Left side - CSV Upload */}
        <div className="w-2/3 h-full">
          <div className="bg-white rounded-lg shadow-lg p-6 flex flex-col">
            <CSVUploadCard/>
          </div>
        </div>

        {/* Right side - Card Grid */}
        <div className="w-1/3 h-full">
          <div className="h-full">
            <CardGrid/>
          </div>
        </div>
      </div>
    </div> 
  );
}