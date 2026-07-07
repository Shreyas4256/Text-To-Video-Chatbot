"use client";

import { forwardRef } from "react";

/* Small shared UI primitives for a consistent look. */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-accent-600 text-white hover:bg-accent-500 disabled:hover:bg-accent-600 shadow-lg shadow-accent-600/20",
  secondary:
    "bg-surface-700 text-zinc-100 hover:bg-surface-600 border border-surface-600",
  ghost: "text-zinc-300 hover:bg-surface-700 hover:text-zinc-100",
  danger: "bg-red-900/60 text-red-100 hover:bg-red-900 border border-red-800/60",
};

export const Button = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
    size?: "sm" | "md" | "lg";
    loading?: boolean;
  }
>(function Button(
  { variant = "primary", size = "md", loading, className = "", children, disabled, ...props },
  ref
) {
  const sizes = {
    sm: "px-2.5 py-1.5 text-xs rounded-md",
    md: "px-4 py-2 text-sm rounded-lg",
    lg: "px-6 py-3 text-base rounded-lg",
  };
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center gap-2 font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-400 disabled:cursor-not-allowed disabled:opacity-60 ${sizes[size]} ${BUTTON_VARIANTS[variant]} ${className}`}
      {...props}
    >
      {loading && <Spinner className="size-4" />}
      {children}
    </button>
  );
});

export const Input = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function Input({ className = "", ...props }, ref) {
  return (
    <input
      ref={ref}
      className={`w-full rounded-lg border border-surface-600 bg-surface-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 transition-colors focus:border-accent-500 focus:outline-none ${className}`}
      {...props}
    />
  );
});

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className = "", ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={`w-full rounded-lg border border-surface-600 bg-surface-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 transition-colors focus:border-accent-500 focus:outline-none ${className}`}
      {...props}
    />
  );
});

export function Select({
  className = "",
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`w-full rounded-lg border border-surface-600 bg-surface-800 px-3 py-2 text-sm text-zinc-100 transition-colors focus:border-accent-500 focus:outline-none ${className}`}
      {...props}
    />
  );
}

export function Label({
  className = "",
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label
      className={`mb-1.5 block text-xs font-medium uppercase tracking-wide text-zinc-400 ${className}`}
      {...props}
    />
  );
}

export function Spinner({ className = "size-5" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
      <path
        fill="currentColor"
        className="opacity-75"
        d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z"
      />
    </svg>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded-lg ${className}`} aria-hidden="true" />;
}

const STATUS_BADGES: Record<string, string> = {
  QUEUED: "bg-amber-950/70 text-amber-300 border-amber-800/50",
  PROCESSING: "bg-sky-950/70 text-sky-300 border-sky-800/50",
  COMPLETED: "bg-emerald-950/70 text-emerald-300 border-emerald-800/50",
  FAILED: "bg-red-950/70 text-red-300 border-red-800/50",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold tracking-wide ${STATUS_BADGES[status] ?? "bg-surface-700 text-zinc-300 border-surface-600"}`}
    >
      {(status === "QUEUED" || status === "PROCESSING") && (
        <span className="relative flex size-1.5">
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60" />
          <span className="relative inline-flex size-1.5 rounded-full bg-current" />
        </span>
      )}
      {status}
    </span>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-surface-600 px-6 py-14 text-center">
      {icon && <div className="mb-3 text-3xl">{icon}</div>}
      <h3 className="text-sm font-semibold text-zinc-200">{title}</h3>
      {description && <p className="mt-1 max-w-sm text-sm text-zinc-500">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
