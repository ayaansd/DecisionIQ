import React, { useState } from "react";
import FileUpload from "./FileUpload";
import AnalysisButtons from "./AnalysisButtons";

export default function Dashboard() {
  const [fileId, setFileId] = useState(null);

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-md p-6">
        <h1 className="text-xl font-bold mb-6">🚀 DECISIONIQ</h1>
        <nav className="space-y-3">
          {[
            "Dashboard",
            "EDA",
            "KPIs",
            "Charts",
            "Summary",
            "Q&A",
            "Goal",
            "Alerts",
          ].map((item) => (
            <button
              key={item}
              className="block w-full text-left text-gray-700 hover:text-blue-600"
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-10">
        <div className="mb-8">
          <FileUpload onFileUploaded={setFileId} />
        </div>
        <AnalysisButtons fileId={fileId} />
      </main>
    </div>
  );
}
