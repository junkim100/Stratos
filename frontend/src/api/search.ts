import { API_CONFIG } from './config';

export interface SearchResponse {
    answer: string;
    sources: string[];
}

export const searchAPI = {
    search: async (query: string): Promise<SearchResponse> => {
        try {
            console.log('Sending request to:', `${API_CONFIG.BASE_URL}/search`);
            console.log('Request payload:', { query });

            const response = await fetch(`${API_CONFIG.BASE_URL}/search`, {
                method: 'POST',
                headers: {
                    ...API_CONFIG.HEADERS,
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'API request failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Search API Error:', error);
            throw error;
        }
    }
};
