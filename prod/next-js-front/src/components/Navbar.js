"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";

export default function Navbar() {
  const pathname = usePathname(); // Get the current path
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // Ensure component is mounted on client side to prevent hydration mismatch
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  // Don't render mobile menu until component is mounted
  if (!isMounted) {
    return (
      <nav className="bg-gradient-to-r from-orange-400 via-orange-500 to-blue-500 fixed w-full z-20 top-0 start-0 shadow-lg">
        <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-0">
          <Link href="/" className="flex items-center space-x-3 rtl:space-x-reverse">
            <img
              src="/logo.png"
              className="h-16"
              alt="Flowbite Logo"
            />
            <span className="self-center text-2xl font-bold whitespace-nowrap text-white tracking-wide">
              <span className="text-orange-200">sc</span>
              <span className="bg-gradient-to-r from-orange-100 to-blue-200 text-transparent bg-clip-text">Drug</span>
              <span className="text-blue-200">Map</span>
            </span>
          </Link>
          <button
            type="button"
            className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-white rounded-lg md:hidden hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white/50"
            aria-controls="navbar-default"
            aria-expanded="false"
          >
            <span className="sr-only">Open main menu</span>
            <svg
              className="w-5 h-5"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 17 14"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M1 1h15M1 7h15M1 13h15"
              />
            </svg>
          </button>
          <div className="hidden w-full md:block md:w-auto" id="navbar-default">
            <ul className="font-medium flex flex-col p-4 md:p-0 mt-4 rounded-lg md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:border-0 bg-transparent">
              <li>
                <Link
                  href="/"
                  className="block py-2 px-3 rounded md:p-0 text-blue-100 hover:text-white transition-colors duration-200"
                >
                  Home
                </Link>
              </li>
              <li>
                <Link
                  href="/readme"
                  className="block py-2 px-3 rounded md:p-0 text-blue-100 hover:text-white transition-colors duration-200"
                >
                  README
                </Link>
              </li>
              <li>
                <Link
                  href="/AnalysisTool"
                  className="block py-2 px-3 rounded md:p-0 text-blue-100 hover:text-white transition-colors duration-200"
                >
                  Tool
                </Link>
              </li>
              <li>
                <Link
                  href="/Data"
                  className="block py-2 px-3 rounded md:p-0 text-blue-100 hover:text-white transition-colors duration-200"
                >
                  Data
                </Link>
              </li>
              <li>
                <Link
                  href="/Contact"
                  className="block py-2 px-3 rounded md:p-0 text-blue-100 hover:text-white transition-colors duration-200"
                >
                  Contact
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    );
  }

  return (
    <nav className="bg-gradient-to-r from-orange-400 via-orange-500 to-blue-500 fixed w-full z-20 top-0 start-0 shadow-lg">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-0">
        <Link href="/" className="flex items-center space-x-3 rtl:space-x-reverse">
          <img
            src="/logo.png"
            className="h-16"
            alt="Flowbite Logo"
          />
          <span className="self-center text-2xl font-bold whitespace-nowrap text-white tracking-wide">
            <span className="text-orange-200">sc</span>
            <span className="bg-gradient-to-r from-orange-100 to-blue-200 text-transparent bg-clip-text">Drug</span>
            <span className="text-blue-200">Map</span>
          </span>
        </Link>
        <button
          onClick={toggleMobileMenu}
          type="button"
          className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-white rounded-lg md:hidden hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white/50"
          aria-controls="navbar-default"
          aria-expanded={isMobileMenuOpen}
        >
          <span className="sr-only">Open main menu</span>
          {isMobileMenuOpen ? (
            // Close icon (X)
            <svg
              className="w-5 h-5"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 14 14"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6"
              />
            </svg>
          ) : (
            // Hamburger menu icon
            <svg
              className="w-5 h-5"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 17 14"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M1 1h15M1 7h15M1 13h15"
              />
            </svg>
          )}
        </button>
        <div className={`${isMobileMenuOpen ? 'block' : 'hidden'} w-full md:block md:w-auto`} id="navbar-default">
          <ul className="font-medium flex flex-col p-4 md:p-0 mt-4 rounded-lg md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:border-0 bg-transparent">
            <li>
              <Link
                href="/"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/" 
                    ? "text-white font-bold" 
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Home
              </Link>
            </li>
            <li>
              <Link
                href="/readme"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/readme"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                README
              </Link>
            </li>
            <li>
              <Link
                href="/AnalysisTool"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/AnalysisTool"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Tool
              </Link>
            </li>
            <li>
              <Link
                href="/Data"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/Data"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Data
              </Link>
            </li>
            <li>
              <Link
                href="/Contact"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/Contact"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Contact
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
