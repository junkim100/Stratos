import React, { useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { ResultCard } from './components/ResultCard';
import { searchAPI, SearchResponse } from './api';
import './styles/App.css';

function App() {
    const [result, setResult] = useState<SearchResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSearch = async (query: string) => {
        setLoading(true);
        setError(null);
        try {
            console.log('Initiating search for:', query);
            const data = await searchAPI.search(query);
            console.log('Search response:', data);
            setResult(data);
        } catch (err) {
            console.error('Search failed:', err);
            setError(err instanceof Error ? err.message : 'Failed to perform search');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <div className="header">
                <h1>Stratos</h1>
            </div>
            <SearchBar onSearch={handleSearch} isLoading={loading} />
            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}
            {loading && (
                <div className="loading-message">
                    Searching...
                </div>
            )}
            {result && <ResultCard result={result} />}
        </div>
    );
}

export default App;
