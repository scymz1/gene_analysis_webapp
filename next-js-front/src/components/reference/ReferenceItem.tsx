type ReferenceProps = {
    authors: string;
    year: string;
    title: string;
    journal: string;
    volumeIssue: string;
    pages: string;
  };
  
  export default function ReferenceItem({
    authors,
    year,
    title,
    journal,
    volumeIssue,
    pages,
  }: ReferenceProps) {
    return (
      <li className="mb-4">
        <p className="text-gray-900 dark:text-gray-100 font-medium">{authors}</p>
        <p className="text-gray-700 dark:text-gray-300">
          {year}. <span className="font-semibold italic">{title}</span>. <span>{journal}</span>, <span>{volumeIssue}</span>, p.{pages}.
        </p>
      </li>
    );
  }
  