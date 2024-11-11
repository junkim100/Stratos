import { useState, FormEvent } from 'react';
import '../styles/SearchBar.css';

interface SearchBarProps {
    onSearch: (query: string) => void;
    isLoading: boolean;
}

export const SearchBar = ({ onSearch, isLoading }: SearchBarProps) => {
    const [query, setQuery] = useState('');

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query);
        }
    };

    return (
        <form className="search-form" onSubmit={handleSubmit}>
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask your question..."
                className="search-input"
                disabled={isLoading}
            />
            <button type="submit" className="search-button" disabled={isLoading}>
                {isLoading ? 'Searching...' : 'Search'}
            </button>
        </form>
    );
};
