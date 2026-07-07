import { NextRequest, NextResponse } from "next/server";

const SESSION_COOKIE = "t2v_session";

/**
 * Edge middleware: fast cookie-presence gate for protected pages.
 * Real session validation (DB-backed) happens in server components and
 * API routes — this only prevents obviously-unauthenticated navigation.
 */
export function middleware(req: NextRequest) {
  const hasSession = Boolean(req.cookies.get(SESSION_COOKIE)?.value);
  const { pathname } = req.nextUrl;

  if (pathname.startsWith("/dashboard") && !hasSession) {
    const url = new URL("/login", req.url);
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }
  if ((pathname === "/login" || pathname === "/signup") && hasSession) {
    return NextResponse.redirect(new URL("/dashboard", req.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/login", "/signup"],
};
