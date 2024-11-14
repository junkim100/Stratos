'use client'

import { useState } from 'react'
import SearchBar from '@/components/SearchBar'
import ResultCard from '@/components/ResultCard'

export default function Home() {
  const [result, setResult] = useState<{ answer: string; sources: string[] } | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSearch = async (query: string) => {
    setLoading(true)
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">Stratos</h1>
      <SearchBar onSearch={handleSearch} isLoading={loading} />
      {loading && <p className="mt-4">Searching...</p>}
      {result && <ResultCard result={result} />}
    </main>
  )
}