import { SearchResponse } from '../api';
import '../styles/ResultCard.css';

interface ResultCardProps {
    result: SearchResponse;
}

export const ResultCard = ({ result }: ResultCardProps) => {
    return (
        <div className="result-card">
            <div className="answer">{result.answer}</div>
            <div className="sources">
                <h4>Sources:</h4>
                <ul>
                    {result.sources.map((source, index) => (
                        <li key={index}>
                            <a href={source} target="_blank" rel="noopener noreferrer">
                                {source}
                            </a>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};
