import type { Metadata, Viewport } from "next";
import "./globals.css";
import { ToastProvider } from "@/components/toast";

export const metadata: Metadata = {
  title: {
    default: "FrameFlow — AI Text-to-Video Chatbot",
    template: "%s · FrameFlow",
  },
  description:
    "Describe a video in plain language and FrameFlow's AI assistant turns it into a real generated video clip.",
};

export const viewport: Viewport = {
  themeColor: "#08080d",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}
