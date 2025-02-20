import CardGrid from "@/components/analysisTool/FlipCard/CardGrid";
import HomePage from "@/components/analysisTool/HomePage";

export default function Home() {
  return (<div className="flex gap-8 h-[calc(100vh-theme(spacing.20))]">
    {/* Left side - CSV Upload */}
    <div className="w-2/3 h-full">
      <div className="bg-white rounded-lg shadow-lg p-6 flex flex-col">
        <HomePage/>
      </div>
    </div>

    {/* Right side - Card Grid */}
    <div className="w-1/3 h-full">
      <div className="h-full">
        <CardGrid/>
      </div>
    </div>
  </div>
  );
}
