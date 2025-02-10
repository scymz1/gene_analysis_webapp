"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname(); // Get the current path

  return (
    <nav className="bg-gradient-to-r from-orange-400 via-orange-500 to-blue-500 fixed w-full z-20 top-0 start-0 shadow-lg">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
        <Link href="/" className="flex items-center space-x-3 rtl:space-x-reverse">
          <img
            src="https://flowbite.com/docs/images/logo.svg"
            className="h-8"
            alt="Flowbite Logo"
          />
          <span className="self-center text-2xl font-semibold whitespace-nowrap text-white">
            Song Lab
          </span>
        </Link>
        <button
          data-collapse-toggle="navbar-default"
          type="button"
          className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-gray-500 rounded-lg md:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600"
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
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/" 
                    ? "text-white font-bold" 
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
              >
                Home
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
              >
                Analysis Tool
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
              >
                Data
              </Link>
            </li>
            <li>
              <Link
                href="/help"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/help"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
              >
                Help
              </Link>
            </li>
            <li>
              <Link
                href="/about"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/about"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
              >
                About
              </Link>
            </li>
            <li>
              <Link
                href="/references"
                className={`block py-2 px-3 rounded md:p-0 ${
                  pathname === "/references"
                    ? "text-white font-bold"
                    : "text-blue-100"
                } hover:text-white transition-colors duration-200`}
              >
                References
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
