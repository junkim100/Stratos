interface ResultCardProps {
  result: {
    answer: string
    sources: string[]
  }
}

export default function ResultCard({ result }: ResultCardProps) {
  return (
    <div className="mt-8 w-full max-w-2xl bg-white shadow-md rounded-lg overflow-hidden">
      <div className="p-4">
        <p className="text-gray-700">{result.answer}</p>
      </div>
      {result.sources.length > 0 && (
        <div className="bg-gray-100 px-4 py-3">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">Sources:</h3>
          <ol className="list-decimal list-inside">
            {result.sources.map((source, index) => (
              <li key={index} className="text-sm text-blue-600 hover:underline">
                <a href={source} target="_blank" rel="noopener noreferrer">
                  {source}
                </a>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}