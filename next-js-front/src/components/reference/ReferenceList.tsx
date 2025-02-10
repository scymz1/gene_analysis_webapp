import ReferenceItem from "./ReferenceItem";

export default function ReferenceList() {
  const references = [
    {
      authors: "Liu, X., Wang, Q., Zhou, M., Wang, Y., Wang, X., Zhou, X., and Song, Q.",
      year: "2024",
      title: "DrugFormer: Graph‐Enhanced Language Model to Predict Drug Sensitivity",
      journal: "Advanced Science",
      volumeIssue: "11(40)",
      pages: "2405861",
    },
    // Add more references as needed
  ];

  return (
    <div className="max-w-screen-md mx-auto mt-10 p-6 bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
        References
      </h2>
      <ul>
        {references.map((ref, index) => (
          <ReferenceItem
            key={index}
            authors={ref.authors}
            year={ref.year}
            title={ref.title}
            journal={ref.journal}
            volumeIssue={ref.volumeIssue}
            pages={ref.pages}
          />
        ))}
      </ul>
    </div>
  );
}
